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
from napari.qt import thread_worker
from napari.utils import Colormap
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .export import export_geojson, export_tiff, export_zarr, save_params
from .pipeline import COMBINE_OPS, CombineParams, combine_masks, remove_small_holes, remove_small_objects
from .resize import lazy_resize
from .rolling_ball import subtract_background, subtract_background_lazy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari
    import napari.layers


# ── UI helpers ────────────────────────────────────────────────────────────────

_FIELD_H = 24  # uniform height for all form field widgets (combo, spin, line edit)


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _form() -> QFormLayout:
    f = QFormLayout()
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    f.setSpacing(5)
    return f


def _dbl_spin(value: float, lo: float, hi: float, decimals: int,
              suffix: str = "", step: float = 0.0) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setValue(value)
    w.setDecimals(decimals)
    if suffix:
        w.setSuffix(f" {suffix}")
    if step:
        w.setSingleStep(step)
    w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    w.setFixedHeight(_FIELD_H)
    return w


def _int_spin(value: int, lo: int, hi: int, suffix: str = "") -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(value)
    if suffix:
        w.setSuffix(f" {suffix}")
    w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    w.setFixedHeight(_FIELD_H)
    return w


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold; margin-top: 4px;")
    return lbl


def _params_to_rows(params: dict) -> list[tuple[str, str]]:
    """Convert a mask_params dict to (label, value) pairs for display."""
    t = params.get("type", "threshold")
    if t == "threshold":
        rows: list[tuple[str, str]] = [
            ("Source layer",    params["source_layer"]),
            ("Source px size",  f"{params['px_size_src_um']} µm/px"),
            ("Mask px size",    f"{params['target_px_size_um']} µm/px"),
            ("Gaussian sigma",  f"{params['gaussian_sigma_px']} px"),
            ("Threshold",       f"{params['threshold']:.1f}"),
            ("Fill holes ≤",    f"{params['hole_threshold_um2']} µm²"),
            ("Remove objects <", f"{params['obj_threshold_um2']} µm²"),
        ]
        if params.get("invert"):
            rows.append(("Colormap", "inverted"))
        return rows
    if t == "combine":
        steps = params.get("steps", [])
        parts = [steps[0]["layer"]] if steps else ["—"]
        for step in steps[1:]:
            parts.append(step["op"])
            parts.append(step["layer"])
        return [
            ("Expression",      " ".join(parts)),
            ("Fill holes ≤",    f"{params['hole_threshold_um2']} µm²"),
            ("Remove objects <", f"{params['obj_threshold_um2']} µm²"),
        ]
    return [(k, str(v)) for k, v in params.items() if k != "type"]


# ── layer list helpers ────────────────────────────────────────────────────────


def _image_layers(viewer: "napari.Viewer") -> list[str]:
    import napari.layers
    return [lyr.name for lyr in reversed(viewer.layers) if isinstance(lyr, napari.layers.Image)]


def _image_layers_no_pre(viewer: "napari.Viewer") -> list[str]:
    import napari.layers
    return [
        lyr.name for lyr in reversed(viewer.layers)
        if isinstance(lyr, napari.layers.Image) and not lyr.name.startswith("pre-")
    ]


def _labels_layers(viewer: "napari.Viewer") -> list[str]:
    import napari.layers
    return [lyr.name for lyr in reversed(viewer.layers) if isinstance(lyr, napari.layers.Labels)]


def _next_label(viewer: "napari.Viewer") -> int:
    """Return the smallest positive label value not already used by any Labels layer."""
    import napari.layers
    used = {
        int(v)
        for lyr in viewer.layers
        if isinstance(lyr, napari.layers.Labels)
        for v in [lyr.data.max()]
        if v > 0
    }
    v = 1
    while v in used:
        v += 1
    return v


def _refresh_combo(combo: QComboBox, choices: list[str]) -> None:
    current = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(choices)
    if current in choices:
        combo.setCurrentText(current)
    combo.blockSignals(False)


# ── spatial helpers ───────────────────────────────────────────────────────────


def _get_layer_data_2d(layer: "napari.layers.Image") -> "np.ndarray | da.Array":
    data = layer.data
    if not isinstance(data, (np.ndarray, da.Array)):
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Layer {layer.name!r} is not 2-D (shape={data.shape})")
    return data


def _compute_mask_transform(
    src_layer: "napari.layers.Image", mask_shape: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float]]:
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
    if invert:
        return Colormap(["black", "black"], name="threshold_inv",
                        low_color="white", high_color="black")
    return Colormap(["white", "white"], name="threshold",
                    low_color="black", high_color="white")


# ── background thread workers ─────────────────────────────────────────────────


@thread_worker
def _rolling_ball_worker(data, radius_px: float, zarr_path: str, num_workers: int):
    return subtract_background(data, radius=radius_px, out_path=zarr_path,
                               num_workers=num_workers)


@thread_worker
def _downsample_worker(src_data, scale: float):
    if not isinstance(src_data, da.Array):
        src_data = da.from_array(src_data)
    return lazy_resize(src_data, scale=scale).compute()


# ── RollingBallWidget ─────────────────────────────────────────────────────────


class RollingBallWidget(QWidget):
    """BG subtraction at source resolution.

    Preview adds a lazy layer instantly; Cache runs in a background thread
    and writes to a zarr on disk with a progress bar.
    Output layer name: {source}-rb-{radius}µm
    """

    def __init__(self, viewer: "napari.Viewer", parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._cache_dir = tempfile.mkdtemp(prefix="masktool_rb_")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── parameters ──
        form = _form()
        self._layer_combo = QComboBox()
        self._layer_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._layer_combo.setFixedHeight(_FIELD_H)
        form.addRow("Image layer:", self._layer_combo)

        self._radius_spin = _dbl_spin(25.0, 1.0, 100_000.0, 1, "µm")
        form.addRow("Ball radius:", self._radius_spin)
        root.addLayout(form)

        root.addWidget(_separator())

        # ── action buttons ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._preview_btn = QPushButton("Preview")
        self._cache_btn = QPushButton("Cache")
        btn_row.addWidget(self._preview_btn)
        btn_row.addWidget(self._cache_btn)
        root.addLayout(btn_row)

        root.addWidget(_separator())

        # ── cache dir row (always visible) ──
        cache_form = _form()
        cache_path_widget = QWidget()
        cache_path_layout = QHBoxLayout(cache_path_widget)
        cache_path_layout.setContentsMargins(0, 0, 0, 0)
        cache_path_layout.setSpacing(4)
        self._cache_path_edit = QLineEdit(self._cache_dir)
        self._cache_path_edit.setReadOnly(True)
        self._cache_path_edit.setFixedHeight(_FIELD_H)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(28)
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(46)
        cache_path_layout.addWidget(self._cache_path_edit)
        cache_path_layout.addWidget(browse_btn)
        cache_path_layout.addWidget(clear_btn)
        cache_form.addRow("Cache dir:", cache_path_widget)
        root.addLayout(cache_form)

        self._preview_btn.clicked.connect(self._on_preview)
        self._cache_btn.clicked.connect(self._on_cache)
        browse_btn.clicked.connect(self._browse_cache_dir)
        clear_btn.clicked.connect(self._clear_cache)

        viewer.layers.events.inserted.connect(lambda _: self._refresh_layers())
        viewer.layers.events.removed.connect(lambda _: self._refresh_layers())
        viewer.layers.selection.events.changed.connect(self._on_selection_changed)
        self._refresh_layers()

    def _refresh_layers(self):
        _refresh_combo(self._layer_combo, _image_layers(self._viewer))

    def _on_selection_changed(self, _event=None):
        import napari.layers
        selected = list(self._viewer.layers.selection)
        if len(selected) != 1:
            return
        layer = selected[0]
        if not isinstance(layer, napari.layers.Image):
            return
        if layer.name in _image_layers(self._viewer):
            self._layer_combo.setCurrentText(layer.name)

    def _current_inputs(self):
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

    def _add_or_replace_image(self, arr, name, scale, translate, **kwargs):
        if name in self._viewer.layers:
            lyr = self._viewer.layers[name]
            lyr.data = arr
            lyr.scale = scale
            lyr.translate = translate
            # kwargs (colormap, contrast_limits) only applied on first creation
        else:
            self._viewer.add_image(arr, name=name, scale=scale, translate=translate, **kwargs)

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
            colormap=layer.colormap,
            contrast_limits=layer.contrast_limits,
        )

    def _on_cache(self):
        inputs = self._current_inputs()
        if inputs is None:
            return
        layer, data, radius_px, out_name = inputs
        zarr_path = str(pathlib.Path(self._cache_dir) / f"{out_name}.zarr")
        scale = tuple(layer.scale[-2:])
        translate = tuple(layer.translate[-2:])
        colormap = layer.colormap
        contrast_limits = layer.contrast_limits

        self._cache_btn.setEnabled(False)
        self._preview_btn.setEnabled(False)

        worker = _rolling_ball_worker(data, radius_px, zarr_path,
                                      num_workers=os.cpu_count() or 1)
        worker.returned.connect(
            lambda arr: self._on_cache_done(arr, out_name, scale, translate,
                                            colormap, contrast_limits)
        )
        worker.errored.connect(self._on_worker_error)
        worker.start()

    def _on_cache_done(self, arr, out_name, scale, translate, colormap, contrast_limits):
        self._cache_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        self._add_or_replace_image(da.from_zarr(arr), out_name, scale, translate,
                                   colormap=colormap, contrast_limits=contrast_limits)

    def _on_worker_error(self, exc):
        self._cache_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        print(f"RollingBallWidget error: {exc}")

    def _browse_cache_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select cache directory", self._cache_dir)
        if d:
            self._cache_dir = d
            self._cache_path_edit.setText(d)

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
        self._preview_raw: np.ndarray | None = None   # downsampled, no blur
        self._preview_data: np.ndarray | None = None  # blurred, used for threshold
        self._src_layer_name: str | None = None
        self._params_log: dict[str, dict] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── parameters ──
        form = _form()

        self._layer_combo = QComboBox()
        self._layer_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._layer_combo.setFixedHeight(_FIELD_H)
        self._layer_combo.currentTextChanged.connect(self._on_layer_changed)
        form.addRow("Image layer:", self._layer_combo)

        self._px_src = _dbl_spin(0.325, 0.001, 100.0, 3, "µm/px")
        form.addRow("Source px size:", self._px_src)

        self._px_tgt = _dbl_spin(10.0, 0.1, 1000.0, 2, "µm/px")
        form.addRow("Mask px size:", self._px_tgt)

        self._sigma = _dbl_spin(1.0, 0.0, 50.0, 1, "px", step=0.5)
        form.addRow("Gaussian sigma:", self._sigma)

        root.addLayout(form)

        from superqt.utils import qdebounced
        self._debounced_sigma = qdebounced(self._apply_sigma, timeout=300, leading=True)
        self._sigma.valueChanged.connect(self._debounced_sigma)

        self._debounced_px = qdebounced(self._on_px_size_changed, timeout=400, leading=True)
        self._px_src.valueChanged.connect(self._debounced_px)
        self._px_tgt.valueChanged.connect(self._debounced_px)

        self._invert = QCheckBox("Invert colormap  (brightfield)")
        self._invert.stateChanged.connect(self._on_invert_changed)
        root.addWidget(self._invert)

        self._add_preview_btn = QPushButton("Add Preview Layer")
        self._add_preview_btn.clicked.connect(self._on_add_preview)
        root.addWidget(self._add_preview_btn)

        root.addWidget(_separator())

        # ── threshold readout + cleanup parameters ──
        form2 = _form()

        self._thresh_label = QLabel("—")
        form2.addRow("Threshold:", self._thresh_label)

        self._holes = _int_spin(1000, 0, 1_000_000_000, "µm²")
        form2.addRow("Fill holes ≤:", self._holes)

        self._objs = _int_spin(0, 0, 1_000_000_000, "µm²")
        form2.addRow("Remove objects <:", self._objs)

        root.addLayout(form2)

        self._finalize_btn = QPushButton("Finalize Mask")
        self._finalize_btn.clicked.connect(self._on_finalize)
        root.addWidget(self._finalize_btn)

        viewer.layers.events.inserted.connect(lambda _: self._refresh_layers())
        viewer.layers.events.removed.connect(self._on_layer_removed)
        self._refresh_layers()
        self._on_layer_changed(self._layer_combo.currentText())

    def _refresh_layers(self):
        _refresh_combo(self._layer_combo, _image_layers_no_pre(self._viewer))

    def _on_layer_changed(self, name: str):
        if name and name in self._viewer.layers:
            self._px_src.setValue(round(float(self._viewer.layers[name].scale[-1]), 6))

    def _on_layer_removed(self, event):
        if self._preview_layer is not None and event.value is self._preview_layer:
            try:
                self._preview_layer.events.contrast_limits.disconnect(self._on_contrast_changed)
            except Exception:
                pass
            self._preview_layer = None
            self._preview_raw = None
            self._preview_data = None
            self._thresh_label.setText("—")
        self._refresh_layers()

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
            if self._preview_data is not None and np.issubdtype(self._preview_data.dtype, np.integer):
                self._thresh_label.setText(str(int(val)))
            else:
                self._thresh_label.setText(f"{val:.1f}")

    def _on_invert_changed(self):
        if self._preview_layer is not None:
            inv = self._invert.isChecked()
            self._preview_layer.colormap = _threshold_colormap(inv)
            self._preview_layer.blending = "translucent" if inv else "additive"

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
        self._src_layer_name = src_name
        self._add_preview_btn.setEnabled(False)

        worker = _downsample_worker(src_data, scale)
        worker.returned.connect(lambda img: self._on_preview_done(img, src_layer))
        worker.errored.connect(self._on_preview_error)
        worker.start()

    def _on_preview_done(self, raw: np.ndarray, src_layer: "napari.layers.Image"):
        self._add_preview_btn.setEnabled(True)
        self._preview_raw = raw
        sigma = self._sigma.value()
        img = cv2.GaussianBlur(raw, ksize=None, sigmaX=float(sigma), sigmaY=float(sigma)) if sigma > 0 else raw.copy()
        self._preview_data = img

        mask_scale, mask_translate = _compute_mask_transform(src_layer, img.shape)
        pre_name = _pre_name(src_layer.name)
        inv = self._invert.isChecked()
        cmap = _threshold_colormap(inv)
        blending = "translucent" if inv else "additive"
        clim = (float(img.min()), float(img.max()))

        if pre_name in self._viewer.layers:
            lyr = self._viewer.layers[pre_name]
            lyr.data = img
            lyr.colormap = cmap
            lyr.scale = mask_scale
            lyr.translate = mask_translate
            # contrast_limits not reset — preserves the user's threshold
            lyr.blending = blending
            lyr.opacity = 0.5
            pre_layer = lyr
        else:
            pre_layer = self._viewer.add_image(
                img, name=pre_name, colormap=cmap,
                scale=mask_scale, translate=mask_translate,
                contrast_limits=clim,
                blending=blending,
                opacity=0.5,
            )

        self._subscribe_preview(pre_layer)

    def _on_px_size_changed(self):
        """Re-downsample when px size changes, if a preview layer exists."""
        if self._preview_raw is not None and self._add_preview_btn.isEnabled():
            self._on_add_preview()

    def _apply_sigma(self):
        """Re-blur the stored raw downsample when sigma changes."""
        if self._preview_raw is None or self._preview_layer is None:
            return
        sigma = self._sigma.value()
        img = cv2.GaussianBlur(self._preview_raw, ksize=None, sigmaX=float(sigma), sigmaY=float(sigma)) if sigma > 0 else self._preview_raw.copy()
        self._preview_data = img
        self._preview_layer.data = img

    def _on_preview_error(self, exc):
        self._add_preview_btn.setEnabled(True)
        print(f"ThresholdWidget preview error: {exc}")

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
            mask.astype(np.uint8) * _next_label(self._viewer), name=fin_name,
            scale=mask_scale, translate=mask_translate,
        )

        mask_params = {
            "type": "threshold",
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
        fin_layer.metadata["mask_params"] = mask_params
        self._params_log[fin_layer.name] = mask_params
        self._preview_layer.visible = False

    def get_params(self) -> dict:
        return dict(self._params_log)


# ── CombineWidget ─────────────────────────────────────────────────────────────


class _MaskRow(QWidget):
    """One row: [op ▾] [layer ▾] [−]"""

    def __init__(
        self,
        viewer: "napari.Viewer",
        first: bool = False,
        filter_shape: tuple | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._viewer = viewer
        self._first = first
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if first:
            lbl = QLabel("Mask 1")
            lbl.setFixedWidth(60)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)
        else:
            self.op_combo = QComboBox()
            self.op_combo.addItems(list(COMBINE_OPS))
            self.op_combo.setFixedWidth(80)
            self.op_combo.setFixedHeight(_FIELD_H)
            layout.addWidget(self.op_combo)

        self.layer_combo = QComboBox()
        self.layer_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.layer_combo.setFixedHeight(_FIELD_H)
        layout.addWidget(self.layer_combo)

        if not first:
            rm_btn = QPushButton("−")
            rm_btn.setFixedWidth(26)
            layout.addWidget(rm_btn)
            self._rm_btn = rm_btn

        self.refresh_layers(filter_shape=filter_shape)

    def refresh_layers(self, filter_shape: tuple | None = None):
        import napari.layers
        current = self.layer_combo.currentData()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for lyr in reversed(self._viewer.layers):
            if not isinstance(lyr, napari.layers.Labels):
                continue
            if filter_shape is not None and lyr.data.shape != filter_shape:
                continue
            scale = lyr.scale[-1]
            self.layer_combo.addItem(f"{lyr.name}  {scale:.2f} µm/px", lyr.name)
        if current is not None:
            idx = self.layer_combo.findData(current)
            if idx >= 0:
                self.layer_combo.setCurrentIndex(idx)
        self.layer_combo.blockSignals(False)

    @property
    def op(self) -> str | None:
        return None if self._first else self.op_combo.currentText()

    @property
    def layer_name(self) -> str:
        return self.layer_combo.currentData() or ""


class CombineWidget(QWidget):
    """Combine 2+ mask layers with logical ops, left-to-right."""

    def __init__(self, viewer: "napari.Viewer", parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._rows: list[_MaskRow] = []
        self._last_params: CombineParams | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.setAlignment(Qt.AlignmentFlag.AlignTop)

        root.addWidget(_section_label("Masks to combine:"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._rows_layout.setSpacing(3)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self._rows_container)
        root.addWidget(scroll)

        add_btn = QPushButton("+ Add mask")
        add_btn.clicked.connect(self._add_row)
        root.addWidget(add_btn)

        root.addWidget(_separator())

        root.addWidget(_section_label("Post-combine cleanup:"))

        form = _form()

        self._holes = _int_spin(0, 0, 1_000_000_000, "µm²")
        form.addRow("Fill holes ≤:", self._holes)

        self._objs = _int_spin(0, 0, 1_000_000_000, "µm²")
        form.addRow("Remove objects <:", self._objs)

        self._out_name = QLineEdit("mask_combined")
        self._out_name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._out_name.setFixedHeight(_FIELD_H)
        form.addRow("Output name:", self._out_name)

        root.addLayout(form)

        compute_btn = QPushButton("Compute")
        compute_btn.clicked.connect(self._on_compute)
        root.addWidget(compute_btn)

        self._add_row(first=True)
        self._add_row()

        viewer.layers.events.inserted.connect(lambda _: self._refresh_all_rows())
        viewer.layers.events.removed.connect(lambda _: self._refresh_all_rows())

    def _seed_shape(self) -> tuple | None:
        if not self._rows:
            return None
        name = self._rows[0].layer_name
        if name and name in self._viewer.layers:
            return self._viewer.layers[name].data.shape
        return None

    def _add_row(self, first: bool = False):
        seed_shape = None if first else self._seed_shape()
        row = _MaskRow(self._viewer, first=first, filter_shape=seed_shape)
        if not first and hasattr(row, "_rm_btn"):
            row._rm_btn.clicked.connect(lambda: self._remove_row(row))
        self._rows.append(row)
        self._rows_layout.addWidget(row)
        if first:
            row.layer_combo.currentIndexChanged.connect(self._on_seed_changed)

    def _on_seed_changed(self):
        for row in self._rows[1:]:
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._rows = self._rows[:1]
        self._add_row()

    def _remove_row(self, row: _MaskRow):
        if len(self._rows) <= 2:
            return
        self._rows.remove(row)
        self._rows_layout.removeWidget(row)
        row.deleteLater()

    def _refresh_all_rows(self):
        seed_shape = self._seed_shape()
        for i, row in enumerate(self._rows):
            row.refresh_layers(filter_shape=None if i == 0 else seed_shape)

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
        hole_px = max(1, int(self._holes.value() / px ** 2)) if self._holes.value() > 0 else 0
        obj_px  = max(1, int(self._objs.value()  / px ** 2)) if self._objs.value()  > 0 else 0

        result = combine_masks(masks, ops, hole_threshold=hole_px, obj_threshold=obj_px)

        steps = [{"layer": layer_names[0]}] + [{"op": op, "layer": ln} for op, ln in zip(ops, layer_names[1:])]
        self._last_params = CombineParams(
            steps=steps,
            hole_threshold=self._holes.value(),
            obj_threshold=self._objs.value(),
        )
        mask_params = {
            "type": "combine",
            "steps": steps,
            "hole_threshold_um2": self._holes.value(),
            "obj_threshold_um2": self._objs.value(),
        }

        name = self._out_name.text().strip() or "mask_combined"
        if name in self._viewer.layers:
            out_layer = self._viewer.layers[name]
            label_val = int(out_layer.data.max()) or 1
            out_layer.data = result.astype(np.uint8) * label_val
        else:
            ref = self._viewer.layers[layer_names[0]]
            out_layer = self._viewer.add_labels(
                result.astype(np.uint8) * _next_label(self._viewer), name=name,
                scale=ref.scale, translate=ref.translate,
            )
        out_layer.metadata["mask_params"] = mask_params

    def get_params(self) -> dict | None:
        return self._last_params.to_dict() if self._last_params else None


# ── ExportWidget ──────────────────────────────────────────────────────────────


class ExportWidget(QWidget):
    """Export a Labels layer as GeoJSON / Zarr / TIFF and write a params JSON sidecar."""

    def __init__(
        self,
        viewer: "napari.Viewer",
        threshold_widget: ThresholdWidget | None = None,
        combine_widget: CombineWidget | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._viewer = viewer
        self._threshold_widget = threshold_widget
        self._combine_widget = combine_widget

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.setAlignment(Qt.AlignmentFlag.AlignTop)

        form = _form()

        self._layer_combo = QComboBox()
        self._layer_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._layer_combo.setFixedHeight(_FIELD_H)
        form.addRow("Mask layer:", self._layer_combo)

        self._px_size = _dbl_spin(10.0, 0.001, 1000.0, 2, "µm/px")
        form.addRow("Pixel size:", self._px_size)

        self._format_combo = QComboBox()
        self._format_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._format_combo.setFixedHeight(_FIELD_H)
        self._format_combo.addItems(["GeoJSON", "GeoJSON (zip)", "Zarr", "TIFF"])
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        form.addRow("Format:", self._format_combo)

        # file path row: line edit + browse button
        path_widget = QWidget()
        path_layout = QHBoxLayout(path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(4)
        self._out_path = QLineEdit(str(pathlib.Path.home() / "mask.geojson"))
        self._out_path.setFixedHeight(_FIELD_H)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(28)
        browse_btn.clicked.connect(self._browse_output)
        path_layout.addWidget(self._out_path)
        path_layout.addWidget(browse_btn)
        form.addRow("Output path:", path_widget)

        root.addLayout(form)

        export_btn = QPushButton("Save mask + log params")
        export_btn.clicked.connect(self._on_export)
        root.addWidget(export_btn)

        self._layer_combo.currentTextChanged.connect(self._on_layer_changed)
        viewer.layers.events.inserted.connect(lambda _: self._refresh_layers())
        viewer.layers.events.removed.connect(lambda _: self._refresh_layers())
        self._refresh_layers()
        self._px_size.setEnabled(self._format_combo.currentText().startswith("GeoJSON"))

    def _refresh_layers(self):
        _refresh_combo(self._layer_combo, _labels_layers(self._viewer))
        self._on_layer_changed(self._layer_combo.currentText())

    def _on_layer_changed(self, name: str):
        if name and name in self._viewer.layers:
            px = float(self._viewer.layers[name].scale[-1])
            self._px_size.setValue(px)

    def _on_format_changed(self, fmt: str):
        path = pathlib.Path(self._out_path.text())
        ext_map = {
            "GeoJSON": ".geojson",
            "GeoJSON (zip)": ".geojson",
            "Zarr": ".zarr",
            "TIFF": ".tiff",
        }
        self._out_path.setText(str(path.with_suffix(ext_map.get(fmt, ".geojson"))))
        self._px_size.setEnabled(fmt.startswith("GeoJSON"))

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save mask", self._out_path.text()
        )
        if path:
            self._out_path.setText(path)

    def _on_export(self):
        layer_name = self._layer_combo.currentText()
        if not layer_name or layer_name not in self._viewer.layers:
            return

        mask = np.asarray(self._viewer.layers[layer_name].data).astype(bool)
        px = self._px_size.value()
        out = pathlib.Path(self._out_path.text())
        fmt = self._format_combo.currentText()

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


# ── MaskInfoWidget ────────────────────────────────────────────────────────────


class MaskInfoWidget(QWidget):
    """Show mask_params metadata for the currently selected Labels layer."""

    def __init__(self, viewer: "napari.Viewer", parent=None):
        super().__init__(parent)
        self._viewer = viewer

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._placeholder = QLabel("Select a mask layer to view its parameters.")
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet("color: gray;")
        root.addWidget(self._placeholder)

        # Grid added directly to root — no QWidget wrapper, which would add
        # implicit style margins and cause extra indentation vs other sections
        self._grid_rows: list[tuple[QLabel, QLabel]] = []
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(5)
        self._grid.setColumnStretch(1, 1)
        root.addLayout(self._grid)

        viewer.layers.selection.events.changed.connect(self._on_selection_changed)
        viewer.layers.events.inserted.connect(self._on_layer_inserted)

    def _on_layer_inserted(self, _event=None):
        # Defer so ThresholdWidget/_on_finalize can set metadata before we read it
        QTimer.singleShot(0, self._on_selection_changed)

    def _on_selection_changed(self, _event=None):
        selected = list(self._viewer.layers.selection)
        if len(selected) != 1:
            self._show_placeholder()
            return
        params = selected[0].metadata.get("mask_params")
        if params is None:
            self._show_placeholder()
            return
        self._show_params(params)

    def _clear_grid(self):
        for lbl, val in self._grid_rows:
            lbl.deleteLater()
            val.deleteLater()
        self._grid_rows.clear()

    def _show_placeholder(self):
        self._clear_grid()
        self._placeholder.setVisible(True)

    def _show_params(self, params: dict):
        self._clear_grid()
        self._placeholder.setVisible(False)
        for i, (label, value) in enumerate(_params_to_rows(params)):
            lbl = QLabel(f"{label}:")
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            val = QLabel(value)
            val.setWordWrap(True)
            self._grid.addWidget(lbl, i, 0)
            self._grid.addWidget(val, i, 1)
            self._grid_rows.append((lbl, val))
