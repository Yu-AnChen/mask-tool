"""
Safe drag-and-drop loading for the mask-builder viewer.

Dropping a file onto napari's default readers bypasses palom: it reads the image
eagerly at full resolution and gives it the wrong pixel size (scale defaults to
1.0), so every physical-units result downstream is wrong. This module installs a
Qt event filter that intercepts drops of recognised image/mask files and routes
them through palom with a small dialog so the user confirms the layer type,
pixel size, and channels before anything is added.

Flow (one file at a time, deferred load):
  drop → construct palom reader for metadata only (lazy) → auto-detect type
  → dialog (all editable) → Add builds layer(s) with correct scale/translate
  → Cancel discards the reader.

Auto-detected type (overridable in the dialog):
  - 3-channel uint8            → RGB  (single (H,W,C) layer)
  - 1-channel                  → mask vs single-channel image, decided by the
                                 foreground adjacent-equality metric (label
                                 images are piecewise-constant)
  - otherwise                  → multi-channel image

Masks are always cached as a nearest-downsampled zarr pyramid (tiny on disk with
zstd) unless the source is already pyramidal. Non-pyramidal single-channel images
larger than 4096 px are cached with INTER_AREA. Everything else uses palom's
levels directly.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import dask.array as da
from numcodecs import Blosc

from qtpy.QtCore import QObject, QEvent, Qt
from qtpy.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox,
    QDoubleSpinBox, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QDialogButtonBox,
)
from napari.qt.threading import thread_worker

try:  # package import (normal) vs. direct execution
    from .pyramid import write_pyramid_group
except ImportError:
    from pyramid import write_pyramid_group


# Extensions palom can open; others fall through to napari's default readers.
_RECOGNISED = (".ome.tif", ".ome.tiff", ".tif", ".tiff", ".qptiff", ".svs", ".vsi")

_EQ_THRESHOLD = 0.5          # foreground-equality above this → mask
_DETECT_WINDOW = 1024        # full-res sample tile edge (px)
_DETECT_MAX_TILES = 6        # cap full-res tiles read during detection
_DETECT_MIN_PAIRS = 100_000  # accumulate foreground pairs up to this
_CACHE_DIM_THRESHOLD = 4096  # non-pyramidal images larger than this get cached


# ── reader / metadata helpers ──────────────────────────────────────────────── #

def _is_recognised(path: str) -> bool:
    return path.lower().endswith(_RECOGNISED)


def _make_reader(path: str, pixel_size: float | None = None):
    import palom.reader as R
    low = path.lower()
    if low.endswith(".svs"):
        return R.SvsReader(path)
    if low.endswith(".vsi"):
        return R.VsiReader(path)
    return R.OmePyramidReader(path, pixel_size=pixel_size)


def _channel_names(path: str, n: int) -> list[str]:
    try:
        import ome_types
        names = [c.name or f"ch{i}"
                 for i, c in enumerate(ome_types.from_tiff(path).images[0].pixels.channels)]
    except Exception:
        names = []
    if len(names) < n:
        names += [f"ch{i}" for i in range(len(names), n)]
    return names[:n]


def _layer_name(path: str) -> str:
    return os.path.basename(path).split(".")[0]


# ── mask-vs-intensity detection ────────────────────────────────────────────── #

def _equal_fg_pairs(a: np.ndarray) -> tuple[int, int]:
    """(equal foreground pairs, total foreground pairs) over right+down neighbours.
    A pair counts only when both pixels are nonzero (background excluded)."""
    nz_r = (a[:, 1:] != 0) & (a[:, :-1] != 0)
    eq_r = nz_r & (a[:, 1:] == a[:, :-1])
    nz_d = (a[1:, :] != 0) & (a[:-1, :] != 0)
    eq_d = nz_d & (a[1:, :] == a[:-1, :])
    return int(eq_r.sum() + eq_d.sum()), int(nz_r.sum() + nz_d.sum())


def _foreground_equality(reader, channel: int = 0) -> float | None:
    """Fraction of neighbouring foreground pixels that are identical, measured on
    full-res sample tiles located via the coarsest pyramid level (densest tissue
    first). Returns None if the content is too sparse to judge."""
    pyr = reader.pyramid
    full = pyr[0][channel]
    H, W = int(full.shape[-2]), int(full.shape[-1])

    # Coarse foreground map to find where the tissue/labels are. The coarse level
    # is only used to *locate* content; the metric is always computed full-res
    # (averaged coarse levels would destroy piecewise-constancy).
    if len(pyr) > 1:
        coarse = np.asarray(pyr[-1][channel])
    else:
        s = max(1, max(H, W) // 2048)
        coarse = np.asarray(full[::s, ::s])

    n_ty = max(1, H // _DETECT_WINDOW)
    n_tx = max(1, W // _DETECT_WINDOW)
    tile_fg = cv2.resize((coarse != 0).astype(np.float32), (n_tx, n_ty),
                         interpolation=cv2.INTER_AREA).ravel()
    order = np.argsort(tile_fg)[::-1]   # densest foreground first

    eq = tot = used = 0
    for idx in order:
        if used >= _DETECT_MAX_TILES or tot >= _DETECT_MIN_PAIRS:
            break
        if tile_fg[idx] <= 0:
            break
        ty, tx = divmod(int(idx), n_tx)
        y0, x0 = ty * _DETECT_WINDOW, tx * _DETECT_WINDOW
        win = np.asarray(full[y0:y0 + _DETECT_WINDOW, x0:x0 + _DETECT_WINDOW])
        e, t = _equal_fg_pairs(win)
        eq += e
        tot += t
        used += 1

    if tot < _DETECT_MIN_PAIRS // 10:
        return None
    return eq / tot


def _detect_type(reader) -> tuple[str, float | None]:
    """Return ("rgb" | "mask" | "image", equality_score_or_None)."""
    p0 = reader.pyramid[0]
    C = int(p0.shape[0])
    dtype = p0.dtype
    if C == 3 and dtype == np.uint8:
        return "rgb", None
    if C == 1:
        score = _foreground_equality(reader)
        if score is None:
            return ("mask" if np.issubdtype(dtype, np.integer) else "image"), None
        return ("mask" if score > _EQ_THRESHOLD else "image"), score
    return "image", None


# ── dialog ─────────────────────────────────────────────────────────────────── #

_TYPE_LABELS = {"image": "Image (multi-channel)", "rgb": "RGB", "mask": "Mask (labels)"}
_TYPE_KEYS = ["image", "rgb", "mask"]


class _AddFileDialog(QDialog):
    def __init__(self, filename, shape, dtype, detected, score, px, ch_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add dropped file")
        self.setModal(True)
        self._ch_names = ch_names

        root = QVBoxLayout(self)
        C, H, W = shape
        info = QLabel(f"<b>{filename}</b><br>{C}×{H}×{W}  ({dtype})")
        info.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(info)

        reason = "—"
        if score is not None:
            reason = (f"label mask — {score:.0%} of neighbouring foreground pixels identical"
                      if detected == "mask"
                      else f"intensity image — only {score:.0%} foreground-equality")
        elif detected == "rgb":
            reason = "3-channel 8-bit → RGB"
        self._reason = QLabel(f"<i>Detected: {reason}</i>")
        self._reason.setTextFormat(Qt.TextFormat.RichText)
        self._reason.setWordWrap(True)
        root.addWidget(self._reason)

        form = QFormLayout()
        self._type = QComboBox()
        for k in _TYPE_KEYS:
            self._type.addItem(_TYPE_LABELS[k], k)
        self._type.setCurrentIndex(_TYPE_KEYS.index(detected))
        self._type.currentIndexChanged.connect(self._on_type_changed)
        form.addRow("Type:", self._type)

        self._px = QDoubleSpinBox()
        self._px.setDecimals(4)
        self._px.setRange(0.0001, 1000.0)
        self._px.setSuffix(" µm/px")
        self._px.setValue(px)
        form.addRow("Pixel size:", self._px)
        root.addLayout(form)

        # Channel section (multi-channel images only) — grows with the dialog.
        self._ch_widget = QWidget()
        ch_layout = QVBoxLayout(self._ch_widget)
        ch_layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(QLabel("Channels:"))
        header.addStretch()
        sel_all = QPushButton("Select all")
        desel_all = QPushButton("Deselect all")
        sel_all.clicked.connect(lambda: self._set_all(Qt.CheckState.Checked))
        desel_all.clicked.connect(lambda: self._set_all(Qt.CheckState.Unchecked))
        header.addWidget(sel_all)
        header.addWidget(desel_all)
        ch_layout.addLayout(header)
        self._ch_list = QListWidget()
        for i, nm in enumerate(ch_names):
            it = QListWidgetItem(f"{i}: {nm}")
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            self._ch_list.addItem(it)
        ch_layout.addWidget(self._ch_list, 1)
        root.addWidget(self._ch_widget, 1)   # take the dialog's extra vertical space

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.resize(380, 460)
        self._on_type_changed()

    def _set_all(self, state):
        for i in range(self._ch_list.count()):
            self._ch_list.item(i).setCheckState(state)

    def _on_type_changed(self):
        self._ch_widget.setVisible(self.layer_type() == "image")

    def layer_type(self) -> str:
        return self._type.currentData()

    def pixel_size(self) -> float:
        return self._px.value()

    def channels(self) -> list[int]:
        sel = [i for i in range(self._ch_list.count())
               if self._ch_list.item(i).checkState() == Qt.CheckState.Checked]
        return sel or [0]


# ── layer construction ─────────────────────────────────────────────────────── #

@thread_worker
def _build_pyramid_worker(level0, out_path, interpolation, compressor):
    return write_pyramid_group(level0, out_path, chunk0=2048, chunk_lo=1024,
                               n_levels=3, factor=4, interpolation=interpolation,
                               dask_workers=os.cpu_count() or 1, compressor=compressor)


def _add_rgb(viewer, reader, px, name):
    pyr = [da.moveaxis(lvl, 0, -1) for lvl in reader.pyramid]   # (H,W,3)
    multiscale = len(pyr) > 1
    lyr = viewer.add_image(pyr if multiscale else pyr[0], name=name, rgb=True,
                           multiscale=multiscale,
                           scale=(px, px), translate=(px / 2, px / 2))
    lyr.metadata["_palom_reader"] = reader


def _add_multichannel(viewer, reader, px, name, channels, ch_names):
    pyr = [lvl[channels] for lvl in reader.pyramid]
    multiscale = len(pyr) > 1
    names = [ch_names[i] for i in channels]
    data = [lvl[::-1] for lvl in pyr] if multiscale else pyr[0][::-1]
    lyrs = viewer.add_image(data, channel_axis=0, name=names[::-1],
                            multiscale=multiscale, contrast_limits=(0, 5000),
                            scale=(px, px), translate=(px / 2, px / 2), visible=False)
    for lyr in (lyrs if isinstance(lyrs, list) else [lyrs]):
        lyr.metadata["_palom_reader"] = reader


def _add_mask_pyramidal(viewer, reader, px, name):
    levels = [lvl[0] for lvl in reader.pyramid]
    if not np.issubdtype(levels[0].dtype, np.integer):
        levels = [lvl.astype(np.int32) for lvl in levels]
    multiscale = len(levels) > 1
    lyr = viewer.add_labels(levels if multiscale else levels[0], name=name,
                            multiscale=multiscale,
                            scale=(px, px), translate=(px / 2, px / 2))
    lyr.metadata["_palom_reader"] = reader


def _cache_and_add(viewer, reader, px, name, *, channel, as_labels, cache_dir):
    level0 = reader.pyramid[0][channel]
    if as_labels and not np.issubdtype(level0.dtype, np.integer):
        level0 = level0.astype(np.int32)
    interp = cv2.INTER_NEAREST if as_labels else cv2.INTER_AREA
    compressor = (Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
                  if as_labels else "default")
    out_path = os.path.join(cache_dir, f"{name}-dnd.zarr")

    def _done(group):
        levels = [da.from_zarr(group[str(i)]) for i in range(len(group))]
        if as_labels:
            viewer.add_labels(levels, name=name, multiscale=True,
                              scale=(px, px), translate=(px / 2, px / 2))
        else:
            viewer.add_image(levels, name=name, multiscale=True,
                             contrast_limits=(0, 5000),
                             scale=(px, px), translate=(px / 2, px / 2))

    worker = _build_pyramid_worker(level0, out_path, interp, compressor)
    worker.returned.connect(_done)
    worker.errored.connect(lambda e: print(f"DnD pyramid build error: {e}"))
    worker.start()


# ── drop handling ──────────────────────────────────────────────────────────── #

def _handle_drop(viewer, path, default_px, cache_dir):
    try:
        reader = _make_reader(path)
    except Exception as e:
        print(f"DnD: could not read {path!r}: {e}")
        return
    try:
        p0 = reader.pyramid[0]
        C, H, W = int(p0.shape[0]), int(p0.shape[-2]), int(p0.shape[-1])
        detected, score = _detect_type(reader)
        file_px = reader.pixel_size
        prefill = file_px if file_px and file_px > 0 else (default_px or 1.0)
        ch_names = _channel_names(path, C)
    except Exception as e:
        print(f"DnD: could not inspect {path!r}: {e}")
        del reader
        return

    dlg = _AddFileDialog(os.path.basename(path), (C, H, W), p0.dtype,
                         detected, score, prefill, ch_names)
    if not dlg.exec():
        del reader   # Cancel → release the file
        return

    typ, px, channels = dlg.layer_type(), dlg.pixel_size(), dlg.channels()
    name = _layer_name(path)
    pyramidal = len(reader.pyramid) > 1

    if typ == "rgb":
        _add_rgb(viewer, reader, px, name)
    elif typ == "mask":
        if pyramidal:
            _add_mask_pyramidal(viewer, reader, px, name)
        else:
            _cache_and_add(viewer, reader, px, name, channel=0,
                           as_labels=True, cache_dir=cache_dir)
    else:  # image
        if not pyramidal and len(channels) == 1 and max(H, W) > _CACHE_DIM_THRESHOLD:
            _cache_and_add(viewer, reader, px, name, channel=channels[0],
                           as_labels=False, cache_dir=cache_dir)
        else:
            _add_multichannel(viewer, reader, px, name, channels, ch_names)


class _DropFilter(QObject):
    def __init__(self, viewer, default_px_size, cache_dir):
        super().__init__()
        self._viewer = viewer
        self._default_px = default_px_size
        self._cache_dir = cache_dir or tempfile.mkdtemp(prefix="masktool_dnd_")

    def eventFilter(self, obj, event):
        et = event.type()
        if et == QEvent.Type.DragEnter:
            urls = event.mimeData().urls()
            if any(_is_recognised(u.toLocalFile()) for u in urls if u.isLocalFile()):
                event.acceptProposedAction()
                return True
            return False
        if et == QEvent.Type.Drop:
            paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            recognised = [p for p in paths if _is_recognised(p)]
            if not recognised:
                return False   # let napari's default readers handle it
            event.acceptProposedAction()
            for p in recognised:
                _handle_drop(self._viewer, p, self._default_px, self._cache_dir)
            return True
        return False


def install_drop_handler(viewer, default_px_size=None, cache_dir=None):
    """Intercept drops of palom-readable files and route them through a dialog.

    Keeps a reference to the filter on the QtViewer so it isn't garbage-collected.
    """
    qtv = viewer.window._qt_viewer
    qtv.setAcceptDrops(True)
    filt = _DropFilter(viewer, default_px_size, cache_dir)
    qtv.installEventFilter(filt)
    qtv._mask_tool_drop_filter = filt   # keep alive
    return filt
