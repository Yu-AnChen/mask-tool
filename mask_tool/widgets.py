"""
napari dock widgets for the mask-building pipeline.

Three panels:
  ChannelMaskWidget  — load channel, resize, BG-subtract, threshold → Labels layer
  CombineWidget      — combine 2+ mask layers with logical ops → Labels layer
  ExportWidget       — export final mask to GeoJSON / Zarr / TIFF + save params
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import zarr
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    FloatSlider,
    FloatSpinBox,
    Label,
    LineEdit,
    PushButton,
    SpinBox,
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .export import export_geojson, export_tiff, export_zarr, save_params
from .pipeline import (
    COMBINE_OPS,
    ChannelMaskParams,
    CombineParams,
    build_channel_mask,
    combine_masks,
)

if TYPE_CHECKING:
    import napari


# ── helpers ──────────────────────────────────────────────────────────────────


def _image_layers(viewer: "napari.Viewer") -> list[str]:
    import napari.layers
    return [lyr.name for lyr in viewer.layers if isinstance(lyr, napari.layers.Image)]


def _labels_layers(viewer: "napari.Viewer") -> list[str]:
    import napari.layers
    return [lyr.name for lyr in viewer.layers if isinstance(lyr, napari.layers.Labels)]


def _get_layer_data_2d(viewer: "napari.Viewer", name: str, channel_idx: int) -> "np.ndarray | da.Array":
    """Extract a single 2-D plane from a napari Image layer."""
    import dask.array as da
    lyr = viewer.layers[name]
    data = lyr.data
    # multiscale: list of arrays — take level 0 (full res)
    if isinstance(data, list):
        data = data[0]
    if isinstance(data, np.ndarray) and data.ndim == 2:
        return data
    if isinstance(data, np.ndarray) and data.ndim == 3:
        return data[channel_idx]
    if isinstance(data, da.Array) and data.ndim == 2:
        return data
    if isinstance(data, da.Array) and data.ndim == 3:
        return data[channel_idx]
    raise ValueError(
        f"Layer {name!r} has unexpected data shape {np.shape(data)}; "
        "expected 2-D or 3-D (C, H, W)."
    )


# ── Channel mask widget ───────────────────────────────────────────────────────


class ChannelMaskWidget(Container):
    """Build a threshold-based binary mask from one channel of an image layer."""

    def __init__(self, viewer: "napari.Viewer"):
        self._viewer = viewer

        # widgets
        self._layer = ComboBox(choices=[], label="Image layer")
        self._channel = SpinBox(value=0, min=0, max=999, label="Channel index")
        self._px_src = FloatSpinBox(value=0.325, min=0.001, max=100.0, step=0.001,
                                    label="Source px size (µm)")
        self._px_tgt = FloatSpinBox(value=10.0, min=0.1, max=1000.0,
                                    label="Target px size (µm)")
        self._bg_sub = CheckBox(value=False, label="Rolling-ball BG subtract")
        self._rb_rad = FloatSpinBox(value=50.0, min=1.0, max=5000.0,
                                    label="  Ball radius (px at target res)")
        self._sigma = FloatSlider(value=1.0, min=0.0, max=20.0, step=0.5,
                                  label="Gaussian sigma", readout=True)
        self._thresh = FloatSlider(value=400.0, min=0.0, max=65535.0, step=10.0,
                                   label="Threshold", readout=True, tracking=True)
        self._holes = SpinBox(value=10, min=0, max=100_000, label="Fill holes ≤ (px²)")
        self._objs = SpinBox(value=10, min=0, max=100_000, label="Remove objects < (px²)")
        self._name = LineEdit(value="mask", label="Output layer name")
        self._btn_build = PushButton(text="Build mask")
        self._btn_apply = PushButton(text="Apply threshold (reuse cache)")
        self._btn_clear = PushButton(text="Clear cache")

        super().__init__(widgets=[
            self._layer, self._channel, self._px_src, self._px_tgt,
            self._bg_sub, self._rb_rad,
            self._sigma, self._thresh,
            self._holes, self._objs,
            self._name,
            self._btn_build, self._btn_apply, self._btn_clear,
        ])

        # cache: {cache_key: zarr.Array}
        self._cache: dict[str, zarr.Array] = {}
        # params log: {cache_key: ChannelMaskParams}
        self._params: dict[str, ChannelMaskParams] = {}

        self._bg_sub.changed.connect(self._on_bg_toggle)
        self._btn_build.changed.connect(self._on_build)
        self._btn_apply.changed.connect(self._on_apply)
        self._btn_clear.changed.connect(self._on_clear)

        self._refresh_layers()
        viewer.layers.events.inserted.connect(lambda _: self._refresh_layers())
        viewer.layers.events.removed.connect(lambda _: self._refresh_layers())

        self._on_bg_toggle(False)

    # ── slots ──

    def _refresh_layers(self):
        choices = _image_layers(self._viewer)
        self._layer.choices = choices
        if choices:
            self._layer.value = choices[0]

    def _on_bg_toggle(self, value):
        self._rb_rad.visible = bool(value)

    def _cache_key(self) -> str:
        return f"{self._layer.value}[ch{self._channel.value}]@{self._px_tgt.value}µm"

    def _on_build(self):
        layer_name = self._layer.value
        if not layer_name:
            return
        src = _get_layer_data_2d(self._viewer, layer_name, self._channel.value)
        params = self._current_params(layer_name)

        cache, mask = build_channel_mask(
            src,
            px_size_src=params.px_size_src,
            target_px_size=params.target_px_size,
            bg_subtract=params.bg_subtract,
            rolling_ball_radius=params.rolling_ball_radius,
            gaussian_sigma=params.gaussian_sigma,
            threshold=params.threshold,
            hole_threshold=params.hole_threshold,
            obj_threshold=params.obj_threshold,
        )
        key = self._cache_key()
        self._cache[key] = cache
        self._params[key] = params
        self._push_labels(mask, params.target_px_size)

    def _on_apply(self):
        """Re-threshold using the existing cache — skips the expensive resize step."""
        key = self._cache_key()
        cache = self._cache.get(key)
        if cache is None:
            print(f"No cache for {key!r} — run 'Build mask' first.")
            return
        layer_name = self._layer.value
        params = self._current_params(layer_name)

        _, mask = build_channel_mask(
            None,  # ignored when existing_cache is provided
            px_size_src=params.px_size_src,
            target_px_size=params.target_px_size,
            gaussian_sigma=params.gaussian_sigma,
            threshold=params.threshold,
            hole_threshold=params.hole_threshold,
            obj_threshold=params.obj_threshold,
            existing_cache=cache,
        )
        self._params[key] = params
        self._push_labels(mask, params.target_px_size)

    def _on_clear(self):
        self._cache.clear()
        self._params.clear()
        print("Cache cleared.")

    # ── helpers ──

    def _current_params(self, layer_name: str) -> ChannelMaskParams:
        return ChannelMaskParams(
            layer_name=layer_name,
            channel_idx=self._channel.value,
            px_size_src=self._px_src.value,
            target_px_size=self._px_tgt.value,
            bg_subtract=self._bg_sub.value,
            rolling_ball_radius=self._rb_rad.value,
            gaussian_sigma=self._sigma.value,
            threshold=self._thresh.value,
            hole_threshold=self._holes.value,
            obj_threshold=self._objs.value,
        )

    def _push_labels(self, mask: np.ndarray, px: float):
        name = self._name.value or self._cache_key()
        scale = (px, px)
        translate = (px * 0.5, px * 0.5)
        if name in self._viewer.layers:
            self._viewer.layers[name].data = mask.astype(np.uint8)
        else:
            self._viewer.add_labels(
                mask.astype(np.uint8),
                name=name,
                scale=scale,
                translate=translate,
            )

    def get_params(self) -> dict:
        return {k: v.to_dict() for k, v in self._params.items()}


# ── Combine widget (dynamic rows, Qt-based) ───────────────────────────────────


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

        # scrollable rows area
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
        self._holes = SpinBox(value=0, min=0, max=100_000, label="Fill holes ≤ (px²)")
        self._objs = SpinBox(value=0, min=0, max=100_000, label="Remove objects < (px²)")
        root.addWidget(self._holes.native)
        root.addWidget(self._objs.native)

        self._out_name = LineEdit(value="mask_combined", label="Output layer name")
        root.addWidget(self._out_name.native)

        compute_btn = QPushButton("Compute")
        compute_btn.clicked.connect(self._on_compute)
        root.addWidget(compute_btn)

        # params log
        self._last_params: CombineParams | None = None

        # seed row + one data row
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
            return  # keep at least seed + one
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

        result = combine_masks(
            masks,
            ops,
            hole_threshold=self._holes.value,
            obj_threshold=self._objs.value,
        )

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
            # inherit scale/translate from the first mask layer
            ref = self._viewer.layers[layer_names[0]]
            self._viewer.add_labels(
                result.astype(np.uint8),
                name=name,
                scale=ref.scale,
                translate=ref.translate,
            )

    def get_params(self) -> dict | None:
        return self._last_params.to_dict() if self._last_params else None


# ── Export widget ─────────────────────────────────────────────────────────────


class ExportWidget(Container):
    """Export a Labels layer as GeoJSON / Zarr / TIFF and write params JSON."""

    def __init__(
        self,
        viewer: "napari.Viewer",
        channel_widget: ChannelMaskWidget | None = None,
        combine_widget: CombineWidget | None = None,
    ):
        self._viewer = viewer
        self._channel_widget = channel_widget
        self._combine_widget = combine_widget

        self._layer = ComboBox(choices=[], label="Mask layer")
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
        self._refresh_layers()
        viewer.layers.events.inserted.connect(lambda _: self._refresh_layers())
        viewer.layers.events.removed.connect(lambda _: self._refresh_layers())

    def _refresh_layers(self):
        choices = _labels_layers(self._viewer)
        self._layer.choices = choices
        if choices:
            self._layer.value = choices[0]

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

        # collect all params and write sidecar JSON
        params: dict = {"output_layer": layer_name, "output_file": str(result)}
        if self._channel_widget:
            params["channel_masks"] = self._channel_widget.get_params()
        if self._combine_widget:
            cp = self._combine_widget.get_params()
            if cp:
                params["combine"] = cp

        log_path = pathlib.Path(str(result)).with_suffix(".params.json")
        save_params(params, log_path)
        print(f"Saved: {result}\nParams: {log_path}")
