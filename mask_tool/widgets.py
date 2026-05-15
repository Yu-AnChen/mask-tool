"""
napari dock widgets for the mask-building tool.

RollingBallWidget  – BG subtraction at source resolution (lazy preview or cached)
ThresholdWidget    – downsample → gaussian → live threshold → finalize mask
CombineWidget      – logical ops on 2+ label layers
ExportWidget       – GeoJSON / Zarr / TIFF export with params log
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import cv2
import dask.array as da
import numpy as np
from magicgui.widgets import (
    ComboBox,
    Container,
    FileEdit,
    FloatSpinBox,
    LineEdit,
    PushButton,
    SpinBox,
)
from napari.qt import thread_worker
from napari.utils import Colormap, progress
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

from .export import export_geojson, export_tiff, export_zarr, save_params
from .pipeline import COMBINE_OPS, CombineParams, combine_masks, remove_small_holes, remove_small_objects
from .resize import lazy_resize
from .rolling_ball import subtract_background, subtract_background_lazy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari
    import napari.layers


# ── module-level helpers ──────────────────────────────────────────────────────


def _image_layers(viewer: "napari.Viewer") -> list[str]:
    import napari.layers
    return [
        lyr.name for lyr in viewer.layers
        if isinstance(lyr, napari.layers.Image)
    ]


def _image_layers_no_pre(viewer: "napari.Viewer") -> list[str]:
    """Image layers excluding pre-{name} preview layers."""
    import napari.layers
    return [
        lyr.name for lyr in viewer.layers
        if isinstance(lyr, napari.layers.Image) and not lyr.name.startswith("pre-")
    ]


def _labels_layers(viewer: "napari.Viewer") -> list[str]:
    import napari.layers
    return [lyr.name for lyr in viewer.layers if isinstance(lyr, napari.layers.Labels)]


def _get_layer_data_2d(layer: "napari.layers.Image") -> "np.ndarray | da.Array":
    """Return the 2-D data array from an Image layer (handles multiscale)."""
    data = layer.data
    if not isinstance(data, (np.ndarray, da.Array)):
        data = data[0]  # multiscale: take full-res level
    if data.ndim != 2:
        raise ValueError(f"Layer {layer.name!r} is not 2-D (shape={data.shape})")
    return data


def _compute_mask_transform(
    src_layer: "napari.layers.Image", mask_shape: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Derive napari scale and translate for a mask from its source image layer.

    Uses world-extent / mask_pixels so the mask aligns precisely regardless of
    per-chunk rounding in lazy_resize.
    """
    data = src_layer.data
    if not isinstance(data, (np.ndarray, da.Array)):
        data = data[0]
    src_H, src_W = data.shape[-2], data.shape[-1]
    sy, sx = float(src_layer.scale[-2]), float(src_layer.scale[-1])
    ty, tx = float(src_layer.translate[-2]), float(src_layer.translate[-1])
    scale_y = src_H * sy / mask_shape[0]
    scale_x = src_W * sx / mask_shape[1]
    tr_y = ty - sy / 2 + scale_y / 2
    tr_x = tx - sx / 2 + scale_x / 2
    return (scale_y, scale_x), (tr_y, tr_x)


def _strip_mask_prefix(name: str) -> str:
    for prefix in ("pre-", "fin-"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _pre_name(src_name: str) -> str:
    return f"pre-{_strip_mask_prefix(src_name)}"


def _fin_name(src_name: str) -> str:
    return f"fin-{_strip_mask_prefix(src_name)}"


def _threshold_colormap(invert: bool = False) -> Colormap:
    """Binary-looking colormap: below contrast_limits[0] = black, above = white."""
    if invert:
        return Colormap(["black", "black"], name="threshold_inv",
                        low_color="white", high_color="black")
    return Colormap(["white", "white"], name="threshold",
                    low_color="black", high_color="white")


# ── background thread workers ────────────────────────────────────────────────


@thread_worker
def _rolling_ball_worker(
    data: "np.ndarray | da.Array",
    radius_px: float,
    zarr_path: str,
    num_workers: int,
):
    with progress(total=0, desc="Rolling ball BG subtraction"):
        return subtract_background(data, radius=radius_px, out_path=zarr_path,
                                   num_workers=num_workers)


@thread_worker
def _downsample_worker(
    src_data: "np.ndarray | da.Array",
    scale: float,
    sigma: float,
):
    if not isinstance(src_data, da.Array):
        src_data = da.from_array(src_data)
    img = lazy_resize(src_data, scale=scale).compute()
    if sigma > 0:
        img = cv2.GaussianBlur(img, ksize=None, sigmaX=float(sigma), sigmaY=float(sigma))
    return img


# ── RollingBallWidget ─────────────────────────────────────────────────────────


class RollingBallWidget(QWidget):
    """BG subtraction at source resolution.

    Preview: lazy dask layer, immediate.
    Cache: computed to zarr in a background thread with progress bar.
    Output layer name: {source}-rb-{radius}µm
    """

    def __init__(self, viewer: "napari.Viewer", parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._cache_dir = tempfile.mkdtemp(prefix="masktool_rb_")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setAlignment(Qt.AlignTop)

        layout.addWidget(QLabel("Image layer:"))
        self._layer_combo = QComboBox()
        layout.addWidget(self._layer_combo)

        form = QFormLayout()
        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(1.0, 100_000.0)
        self._radius_spin.setValue(25.0)
        self._radius_spin.setDecimals(1)
        self._radius_spin.setSuffix(" µm")
        form.addRow("Ball radius:", self._radius_spin)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self._preview_btn = QPushButton("Preview")
        self._cache_btn = QPushButton("Cache")
        btn_row.addWidget(self._preview_btn)
        btn_row.addWidget(self._cache_btn)
        layout.addLayout(btn_row)

        # ── collapsible cache settings ──
        collapsible = QCollapsible("Cache settings", self)
        cache_inner = QWidget()
        cache_layout = QVBoxLayout(cache_inner)
        cache_layout.setContentsMargins(4, 4, 4, 4)

        self._cache_dir_label = QLabel(self._cache_dir)
        self._cache_dir_label.setWordWrap(True)
        cache_layout.addWidget(QLabel("Cache directory:"))
        cache_layout.addWidget(self._cache_dir_label)

        dir_row = QHBoxLayout()
        browse_btn = QPushButton("Browse…")
        clear_btn = QPushButton("Clear cache")
        dir_row.addWidget(browse_btn)
        dir_row.addWidget(clear_btn)
        cache_layout.addLayout(dir_row)

        collapsible.addWidget(cache_inner)
        layout.addWidget(collapsible)

        self._preview_btn.clicked.connect(self._on_preview)
        self._cache_btn.clicked.connect(self._on_cache)
        browse_btn.clicked.connect(self._browse_cache_dir)
        clear_btn.clicked.connect(self._clear_cache)

        viewer.layers.events.inserted.connect(lambda _: self._refresh_layers())
        viewer.layers.events.removed.connect(lambda _: self._refresh_layers())
        self._refresh_layers()

    # ── layer list management ──

    def _refresh_layers(self):
        current = self._layer_combo.currentText()
        self._layer_combo.blockSignals(True)
        self._layer_combo.clear()
        self._layer_combo.addItems(_image_layers(self._viewer))
        if current in _image_layers(self._viewer):
            self._layer_combo.setCurrentText(current)
        self._layer_combo.blockSignals(False)

    # ── helpers ──

    def _current_inputs(self):
        """Return (layer, data, radius_px, out_name) or None."""
        name = self._layer_combo.currentText()
        if not name or name not in self._viewer.layers:
            return None
        layer = self._viewer.layers[name]
        try:
            data = _get_layer_data_2d(layer)
        except ValueError as e:
            print(f"RollingBallWidget: {e}")
            return None
        radius_µm = self._radius_spin.value()
        radius_px = radius_µm / float(layer.scale[-1])
        out_name = f"{name}-rb-{radius_µm:.0f}µm"
        return layer, data, radius_px, out_name

    def _add_or_replace_image(self, arr, name, scale, translate):
        if name in self._viewer.layers:
            lyr = self._viewer.layers[name]
            lyr.data = arr
            lyr.scale = scale
            lyr.translate = translate
        else:
            self._viewer.add_image(arr, name=name, scale=scale, translate=translate)

    # ── button handlers ──

    def _on_preview(self):
        inputs = self._current_inputs()
        if inputs is None:
            return
        layer, data, radius_px, out_name = inputs
        lazy_result = subtract_background_lazy(data, radius=radius_px)
        self._add_or_replace_image(
            lazy_result, out_name,
            scale=tuple(layer.scale[-2:]),
            translate=tuple(layer.translate[-2:]),
        )

    def _on_cache(self):
        inputs = self._current_inputs()
        if inputs is None:
            return
        layer, data, radius_px, out_name = inputs
        zarr_path = str(pathlib.Path(self._cache_dir) / f"{out_name}.zarr")
        scale = tuple(layer.scale[-2:])
        translate = tuple(layer.translate[-2:])

        self._cache_btn.setEnabled(False)
        self._preview_btn.setEnabled(False)

        worker = _rolling_ball_worker(data, radius_px, zarr_path,
                                      num_workers=os.cpu_count() or 1)
        worker.returned.connect(
            lambda arr: self._on_cache_done(arr, out_name, scale, translate)
        )
        worker.errored.connect(self._on_worker_error)
        worker.start()

    def _on_cache_done(self, arr, out_name, scale, translate):
        self._cache_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        dask_arr = da.from_zarr(arr)
        self._add_or_replace_image(dask_arr, out_name, scale, translate)

    def _on_worker_error(self, exc):
        self._cache_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        print(f"RollingBallWidget error: {exc}")

    def _browse_cache_dir(self):
        from qtpy.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(self, "Select cache directory", self._cache_dir)
        if d:
            self._cache_dir = d
            self._cache_dir_label.setText(d)

    def _clear_cache(self):
        import shutil
        for p in pathlib.Path(self._cache_dir).glob("*.zarr"):
            shutil.rmtree(p, ignore_errors=True)
        print(f"Cache cleared: {self._cache_dir}")


# ── ThresholdWidget ───────────────────────────────────────────────────────────


class ThresholdWidget(QWidget):
    """Downsample → gaussian → live threshold preview → finalize to Labels layer.

    Layer naming:
      preview image layer : pre-{source_base_name}
      final labels layer  : fin-{source_base_name}  (accumulates; no overwrite)
    """

    def __init__(self, viewer: "napari.Viewer", parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._preview_layer: "napari.layers.Image | None" = None
        self._preview_data: np.ndarray | None = None
        self._src_layer_name: str | None = None
        self._params_log: dict[str, dict] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setAlignment(Qt.AlignTop)

        layout.addWidget(QLabel("Image layer:"))
        self._layer_combo = QComboBox()
        self._layer_combo.currentTextChanged.connect(self._on_layer_changed)
        layout.addWidget(self._layer_combo)

        form = QFormLayout()

        self._px_src = QDoubleSpinBox()
        self._px_src.setRange(0.001, 100.0)
        self._px_src.setValue(0.325)
        self._px_src.setDecimals(4)
        self._px_src.setSuffix(" µm/px")
        form.addRow("Source px size:", self._px_src)

        self._px_tgt = QDoubleSpinBox()
        self._px_tgt.setRange(0.1, 1000.0)
        self._px_tgt.setValue(10.0)
        self._px_tgt.setDecimals(2)
        self._px_tgt.setSuffix(" µm/px")
        form.addRow("Mask px size:", self._px_tgt)

        self._sigma = QDoubleSpinBox()
        self._sigma.setRange(0.0, 50.0)
        self._sigma.setValue(1.0)
        self._sigma.setDecimals(1)
        self._sigma.setSuffix(" px")
        form.addRow("Gaussian sigma:", self._sigma)

        layout.addLayout(form)

        self._invert = QCheckBox("Invert colormap (brightfield)")
        self._invert.stateChanged.connect(self._on_invert_changed)
        layout.addWidget(self._invert)

        self._add_preview_btn = QPushButton("Add Preview Layer")
        self._add_preview_btn.clicked.connect(self._on_add_preview)
        layout.addWidget(self._add_preview_btn)

        sep = QLabel()
        sep.setFrameShape(QLabel.HLine if hasattr(QLabel, "HLine") else 0)
        layout.addWidget(sep)

        self._thresh_label = QLabel("Threshold: —")
        layout.addWidget(self._thresh_label)

        form2 = QFormLayout()

        self._holes = QSpinBox()
        self._holes.setRange(0, 1_000_000_000)
        self._holes.setValue(1000)
        self._holes.setSuffix(" µm²")
        form2.addRow("Fill holes ≤:", self._holes)

        self._objs = QSpinBox()
        self._objs.setRange(0, 1_000_000_000)
        self._objs.setValue(1000)
        self._objs.setSuffix(" µm²")
        form2.addRow("Remove objects <:", self._objs)

        layout.addLayout(form2)

        self._finalize_btn = QPushButton("Finalize Mask")
        self._finalize_btn.clicked.connect(self._on_finalize)
        layout.addWidget(self._finalize_btn)

        viewer.layers.events.inserted.connect(lambda _: self._refresh_layers())
        viewer.layers.events.removed.connect(self._on_layer_removed)
        self._refresh_layers()

    # ── layer list management ──

    def _refresh_layers(self):
        current = self._layer_combo.currentText()
        self._layer_combo.blockSignals(True)
        self._layer_combo.clear()
        choices = _image_layers_no_pre(self._viewer)
        self._layer_combo.addItems(choices)
        if current in choices:
            self._layer_combo.setCurrentText(current)
        self._layer_combo.blockSignals(False)

    def _on_layer_changed(self, name: str):
        if not name or name not in self._viewer.layers:
            return
        layer = self._viewer.layers[name]
        self._px_src.setValue(round(float(layer.scale[-1]), 6))

    def _on_layer_removed(self, event):
        removed = event.value
        if self._preview_layer is not None and removed is self._preview_layer:
            try:
                self._preview_layer.events.contrast_limits.disconnect(self._on_contrast_changed)
            except Exception:
                pass
            self._preview_layer = None
            self._preview_data = None
            self._thresh_label.setText("Threshold: —")
        self._refresh_layers()

    # ── contrast limit subscription ──

    def _subscribe_preview(self, layer: "napari.layers.Image"):
        if self._preview_layer is not None:
            try:
                self._preview_layer.events.contrast_limits.disconnect(self._on_contrast_changed)
            except Exception:
                pass
        self._preview_layer = layer
        layer.events.contrast_limits.connect(self._on_contrast_changed)
        self._on_contrast_changed()

    def _on_contrast_changed(self):
        if self._preview_layer is not None:
            val = self._preview_layer.contrast_limits[0]
            self._thresh_label.setText(f"Threshold: {val:.1f}")

    def _on_invert_changed(self):
        if self._preview_layer is not None:
            self._preview_layer.colormap = _threshold_colormap(self._invert.isChecked())

    # ── Add Preview Layer ──

    def _on_add_preview(self):
        src_name = self._layer_combo.currentText()
        if not src_name or src_name not in self._viewer.layers:
            return
        src_layer = self._viewer.layers[src_name]
        try:
            src_data = _get_layer_data_2d(src_layer)
        except ValueError as e:
            print(f"ThresholdWidget: {e}")
            return

        scale = self._px_src.value() / self._px_tgt.value()
        sigma = self._sigma.value()
        self._src_layer_name = src_name

        self._add_preview_btn.setEnabled(False)

        worker = _downsample_worker(src_data, scale, sigma)
        worker.returned.connect(lambda img: self._on_preview_done(img, src_layer))
        worker.errored.connect(self._on_preview_error)
        worker.start()

    def _on_preview_done(self, img: np.ndarray, src_layer: "napari.layers.Image"):
        self._add_preview_btn.setEnabled(True)
        self._preview_data = img

        mask_scale, mask_translate = _compute_mask_transform(src_layer, img.shape)
        pre_name = _pre_name(src_layer.name)
        cmap = _threshold_colormap(self._invert.isChecked())

        clim = (float(img.min()), float(img.max()))

        if pre_name in self._viewer.layers:
            lyr = self._viewer.layers[pre_name]
            lyr.data = img
            lyr.colormap = cmap
            lyr.scale = mask_scale
            lyr.translate = mask_translate
            lyr.contrast_limits = clim
            pre_layer = lyr
        else:
            pre_layer = self._viewer.add_image(
                img, name=pre_name, colormap=cmap,
                scale=mask_scale, translate=mask_translate,
                contrast_limits=clim,
            )

        self._subscribe_preview(pre_layer)

    def _on_preview_error(self, exc):
        self._add_preview_btn.setEnabled(True)
        print(f"ThresholdWidget preview error: {exc}")

    # ── Finalize Mask ──

    def _on_finalize(self):
        if self._preview_layer is None or self._preview_data is None:
            print("No preview layer — click 'Add Preview Layer' first.")
            return
        if self._src_layer_name not in self._viewer.layers:
            print("Source layer no longer exists.")
            return

        threshold = self._preview_layer.contrast_limits[0]
        px_tgt = self._px_tgt.value()

        hole_px = max(1, int(self._holes.value() / px_tgt ** 2)) if self._holes.value() > 0 else 0
        obj_px  = max(1, int(self._objs.value()  / px_tgt ** 2)) if self._objs.value()  > 0 else 0

        mask = self._preview_data > threshold
        if hole_px > 0:
            mask = remove_small_holes(mask, hole_px, connectivity=2)
        if obj_px > 0:
            mask = remove_small_objects(mask, obj_px, connectivity=2)

        src_layer = self._viewer.layers[self._src_layer_name]
        mask_scale, mask_translate = _compute_mask_transform(src_layer, mask.shape)

        fin_name = _fin_name(self._src_layer_name)
        fin_layer = self._viewer.add_labels(
            mask.astype(np.uint8),
            name=fin_name,
            scale=mask_scale,
            translate=mask_translate,
        )

        self._params_log[fin_layer.name] = {
            "source_layer": self._src_layer_name,
            "preview_layer": self._preview_layer.name,
            "px_size_src_um": self._px_src.value(),
            "target_px_size_um": px_tgt,
            "gaussian_sigma_px": self._sigma.value(),
            "threshold": threshold,
            "hole_threshold_um2": self._holes.value(),
            "obj_threshold_um2": self._objs.value(),
            "invert": self._invert.isChecked(),
        }

        self._preview_layer.visible = False

    def get_params(self) -> dict:
        return dict(self._params_log)


# ── CombineWidget ─────────────────────────────────────────────────────────────


class _MaskRow(QWidget):
    """One row in the combine list: [op ▾] [layer ▾] [−]"""

    def __init__(self, viewer: "napari.Viewer", first: bool = False, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if first:
            lbl = QLabel("seed")
            lbl.setFixedWidth(70)
            layout.addWidget(lbl)
        else:
            self.op_combo = QComboBox()
            self.op_combo.addItems(list(COMBINE_OPS))
            self.op_combo.setFixedWidth(80)
            layout.addWidget(self.op_combo)

        self.layer_combo = QComboBox()
        self.layer_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.layer_combo)

        if not first:
            rm_btn = QPushButton("−")
            rm_btn.setFixedWidth(24)
            layout.addWidget(rm_btn)
            self._rm_btn = rm_btn

        self._first = first
        self.refresh_layers()

    def refresh_layers(self):
        current = self.layer_combo.currentText()
        self.layer_combo.clear()
        choices = _labels_layers(self._viewer)
        self.layer_combo.addItems(choices)
        if current in choices:
            self.layer_combo.setCurrentText(current)

    @property
    def op(self) -> str | None:
        return None if self._first else self.op_combo.currentText()

    @property
    def layer_name(self) -> str:
        return self.layer_combo.currentText()


class CombineWidget(QWidget):
    """Combine 2+ mask layers with logical ops, left-to-right."""

    def __init__(self, viewer: "napari.Viewer", parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._rows: list[_MaskRow] = []

        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(250)
        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(self._rows_container)
        root.addWidget(QLabel("Masks to combine:"))
        root.addWidget(scroll)

        add_btn = QPushButton("+ Add layer")
        add_btn.clicked.connect(self._add_row)
        root.addWidget(add_btn)

        root.addWidget(QLabel("Post-combine cleanup:"))
        self._holes = SpinBox(value=0, min=0, max=100_000_000, label="Fill holes ≤ (µm²)")
        self._objs  = SpinBox(value=0, min=0, max=100_000_000, label="Remove objects < (µm²)")
        root.addWidget(self._holes.native)
        root.addWidget(self._objs.native)

        self._out_name = LineEdit(value="mask_combined", label="Output layer name")
        root.addWidget(self._out_name.native)

        compute_btn = QPushButton("Compute")
        compute_btn.clicked.connect(self._on_compute)
        root.addWidget(compute_btn)

        self._last_params: CombineParams | None = None

        self._add_row(first=True)
        self._add_row()

        viewer.layers.events.inserted.connect(lambda _: self._refresh_all_rows())
        viewer.layers.events.removed.connect(lambda _: self._refresh_all_rows())

    def _add_row(self, first: bool = False):
        row = _MaskRow(self._viewer, first=first)
        if not first and hasattr(row, "_rm_btn"):
            row._rm_btn.clicked.connect(lambda: self._remove_row(row))
        self._rows.append(row)
        self._rows_layout.addWidget(row)

    def _remove_row(self, row: _MaskRow):
        if len(self._rows) <= 2:
            return
        self._rows.remove(row)
        self._rows_layout.removeWidget(row)
        row.deleteLater()

    def _refresh_all_rows(self):
        for row in self._rows:
            row.refresh_layers()

    def _on_compute(self):
        layer_names = [r.layer_name for r in self._rows]
        ops = [r.op for r in self._rows if r.op is not None]

        masks = []
        for name in layer_names:
            if not name or name not in self._viewer.layers:
                print(f"Layer {name!r} not found — skipping combine.")
                return
            masks.append(np.asarray(self._viewer.layers[name].data).astype(bool))

        px = float(self._viewer.layers[layer_names[0]].scale[-1])
        hole_px = max(1, int(self._holes.value / px ** 2)) if self._holes.value > 0 else 0
        obj_px  = max(1, int(self._objs.value  / px ** 2)) if self._objs.value  > 0 else 0

        result = combine_masks(masks, ops, hole_threshold=hole_px, obj_threshold=obj_px)

        self._last_params = CombineParams(
            steps=[{"layer": layer_names[0]}]
                  + [{"op": op, "layer": ln} for op, ln in zip(ops, layer_names[1:])],
            hole_threshold=self._holes.value,
            obj_threshold=self._objs.value,
        )

        name = self._out_name.value or "mask_combined"
        if name in self._viewer.layers:
            self._viewer.layers[name].data = result.astype(np.uint8)
        else:
            ref = self._viewer.layers[layer_names[0]]
            self._viewer.add_labels(
                result.astype(np.uint8), name=name,
                scale=ref.scale, translate=ref.translate,
            )

    def get_params(self) -> dict | None:
        return self._last_params.to_dict() if self._last_params else None


# ── ExportWidget ──────────────────────────────────────────────────────────────


class ExportWidget(Container):
    """Export a Labels layer as GeoJSON / Zarr / TIFF and write params JSON."""

    def __init__(
        self,
        viewer: "napari.Viewer",
        threshold_widget: ThresholdWidget | None = None,
        combine_widget: CombineWidget | None = None,
    ):
        self._viewer = viewer
        self._threshold_widget = threshold_widget
        self._combine_widget = combine_widget

        self._layer = ComboBox(
            choices=lambda _w: _labels_layers(viewer),
            label="Mask layer",
        )
        self._px_size = FloatSpinBox(value=10.0, min=0.001, max=1000.0,
                                     label="Pixel size (µm)")
        self._format = ComboBox(
            choices=["GeoJSON", "GeoJSON (zip)", "Zarr", "TIFF"],
            label="Format",
        )
        self._out_path = FileEdit(
            value=pathlib.Path.home() / "mask.geojson",
            label="Output path",
            mode="w",
        )
        self._btn_export = PushButton(text="Save mask + log params")

        super().__init__(widgets=[
            self._layer, self._px_size, self._format, self._out_path, self._btn_export
        ])

        self._btn_export.changed.connect(self._on_export)
        viewer.layers.events.inserted.connect(lambda _: self._layer.reset_choices())
        viewer.layers.events.removed.connect(lambda _: self._layer.reset_choices())

    def _on_export(self):
        layer_name = self._layer.value
        if not layer_name:
            return

        mask = np.asarray(self._viewer.layers[layer_name].data).astype(bool)
        px = self._px_size.value
        out = pathlib.Path(str(self._out_path.value))
        fmt = self._format.value

        if fmt == "GeoJSON":
            result = export_geojson(mask, px, out.with_suffix(".geojson"))
        elif fmt == "GeoJSON (zip)":
            result = export_geojson(mask, px, out.with_suffix(".geojson"), compress=True)
        elif fmt == "Zarr":
            result = export_zarr(mask, out, pixel_size=px)
        else:
            result = export_tiff(mask, out.with_suffix(".tiff"), pixel_size=px)

        params: dict = {"output_layer": layer_name, "output_file": str(result)}
        if self._threshold_widget:
            params["threshold_masks"] = self._threshold_widget.get_params()
        if self._combine_widget:
            cp = self._combine_widget.get_params()
            if cp:
                params["combine"] = cp

        log_path = pathlib.Path(str(result)).with_suffix(".params.json")
        save_params(params, log_path)
        print(f"Saved: {result}\nParams: {log_path}")
