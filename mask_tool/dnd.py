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

Non-pyramidal masks/images are cached: the reader's native level 0 stays on top
and only the cheap coarse levels are built (nearest for masks — strided, so uint32
labels work; INTER_AREA for images) into an in-memory zstd zarr (tens of MB, lives
with the layer, no disk cache). Truly pyramidal sources reuse their stored levels.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import dask.array as da
from numcodecs import Blosc

from qtpy.QtCore import QObject, QEvent, Qt
from qtpy.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox,
    QDoubleSpinBox, QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem,
    QDialogButtonBox,
)
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

try:  # package import (normal) vs. direct execution
    from .pyramid import write_pyramid_group
except ImportError:
    from pyramid import write_pyramid_group

# Names whose pyramid is currently building in the background — guards against an
# impatient user re-dropping the same file and starting duplicate builds.
_BUILDING: set = set()


# Extensions palom can open; others fall through to napari's default readers.
_RECOGNISED = (".ome.tif", ".ome.tiff", ".tif", ".tiff", ".qptiff", ".svs", ".vsi")

_EQ_THRESHOLD = 0.5          # foreground-equality above this → mask
_DETECT_WINDOW = 1024        # full-res sample tile edge (px)
_DETECT_MAX_TILES = 6        # cap full-res tiles read during detection
_DETECT_MIN_PAIRS = 100_000  # accumulate foreground pairs up to this
_CACHE_DIM_THRESHOLD = 4096  # non-pyramidal images larger than this get cached
_CACHE_WORKERS = 4           # dask threads for the cache build; capped because the
                             # downsample is I/O-bound, and peak RAM ∝ workers
_PYRAMID_FACTOR = 2          # downsample factor between cached coarse levels
                             # (2 = finer steps / smoother zoom; 4 = fewer levels)
_PYRAMID_MIN_DIM = 1024      # build coarse levels until the coarsest max-dim ≤ this


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


def _is_pyramidal(path: str) -> bool:
    """True if the file stores real pyramid levels (mirrors palom's own check).

    palom's OmePyramidReader uses stored levels when present, but otherwise falls
    back to synthesising a pyramid by coarsening level 0 (da.coarsen) — in which
    case `reader.pyramid` still has >1 level, but every coarse level re-reads the
    full-res level 0. So `len(reader.pyramid) > 1` cannot tell the two apart; we
    check the source directly.
    """
    low = path.lower()
    if low.endswith((".svs", ".vsi")):
        return True   # these readers expose real pyramid levels
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            return len(tf.series) > 1 or len(tf.series[0].levels) > 1
    except Exception:
        return False


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


class _ChannelList(QListWidget):
    """Checkable list where a left-click anywhere on a row toggles its checkbox
    (not just the indicator), toggling exactly once."""

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is not None and event.button() == Qt.MouseButton.LeftButton:
            new = (Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked
                   else Qt.CheckState.Checked)
            item.setCheckState(new)
            self.setCurrentItem(item)
            return
        super().mousePressEvent(event)


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
        self._ch_filter = QLineEdit()
        self._ch_filter.setPlaceholderText("Filter channels…")
        self._ch_filter.setClearButtonEnabled(True)
        self._ch_filter.textChanged.connect(self._apply_filter)
        ch_layout.addWidget(self._ch_filter)
        self._ch_list = _ChannelList()
        self._ch_list.setMinimumHeight(240)
        for i, nm in enumerate(ch_names):
            it = QListWidgetItem(f"{i}: {nm}")
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked)
            self._ch_list.addItem(it)
        ch_layout.addWidget(self._ch_list, 1)
        root.addWidget(self._ch_widget, 1)   # take the dialog's extra vertical space

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.setMinimumWidth(400)
        self._on_type_changed()

    def _set_all(self, state):
        # Act on visible (filter-matching) items only, so a filter + Select all
        # checks just the matches.
        for i in range(self._ch_list.count()):
            it = self._ch_list.item(i)
            if not it.isHidden():
                it.setCheckState(state)

    def _apply_filter(self, text):
        t = text.lower()
        for i in range(self._ch_list.count()):
            it = self._ch_list.item(i)
            it.setHidden(t not in it.text().lower())

    def _on_type_changed(self):
        # Channel list only applies to multi-channel images; hide it otherwise
        # and shrink the dialog to fit so there are no empty gaps.
        self._ch_widget.setVisible(self.layer_type() == "image")
        self.adjustSize()

    def layer_type(self) -> str:
        return self._type.currentData()

    def pixel_size(self) -> float:
        return self._px.value()

    def channels(self) -> list[int]:
        sel = [i for i in range(self._ch_list.count())
               if self._ch_list.item(i).checkState() == Qt.CheckState.Checked]
        return sel or [0]


# ── layer construction ─────────────────────────────────────────────────────── #

def _n_levels(shape, factor: int, min_dim: int) -> int:
    """Levels (incl. level 0) so the coarsest level's max dim is ≤ min_dim."""
    m = max(int(shape[-2]), int(shape[-1]))
    n = 1
    while m > min_dim:
        m = -(-m // factor)   # ceil div
        n += 1
    return max(n, 2)          # at least one coarse level


@thread_worker
def _build_pyramid_worker(level0, interpolation, compressor):
    # Build only the cheap coarse levels into an in-memory (zstd) zarr — the
    # caller keeps the reader's native level 0 for the top of the multiscale
    # stack. The full-res level 0 is read once to make level 1; we skip
    # re-compressing/re-reading and any disk cache. Coarse levels are tiny
    # (~tens of MB compressed), so they live in RAM with the layer. Build enough
    # levels (at _PYRAMID_FACTOR steps) to reach a small coarsest level, so napari
    # has a cheap level for every zoom and the overview isn't a huge raster.
    workers = min(_CACHE_WORKERS, os.cpu_count() or 1)
    n_levels = _n_levels(level0.shape, _PYRAMID_FACTOR, _PYRAMID_MIN_DIM)
    return write_pyramid_group(level0, None, chunk0=2048, chunk_lo=1024,
                               n_levels=n_levels, factor=_PYRAMID_FACTOR,
                               interpolation=interpolation,
                               dask_workers=workers, compressor=compressor,
                               store_level0=False)


def _add_rgb(viewer, reader, px, name):
    pyr = [da.moveaxis(lvl, 0, -1) for lvl in reader.pyramid]   # (H,W,3)
    multiscale = len(pyr) > 1
    lyr = viewer.add_image(pyr if multiscale else pyr[0], name=name, rgb=True,
                           multiscale=multiscale,
                           scale=(px, px), translate=(px / 2, px / 2))
    lyr.metadata["_palom_reader"] = reader


def _add_multichannel(viewer, reader, px, channels, ch_names):
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
    """Reuse the file's real stored pyramid levels (cheap coarse reads)."""
    levels = [lvl[0] for lvl in reader.pyramid]
    if not np.issubdtype(levels[0].dtype, np.integer):
        levels = [lvl.astype(np.int32) for lvl in levels]
    multiscale = len(levels) > 1
    lyr = viewer.add_labels(levels if multiscale else levels[0], name=name,
                            multiscale=multiscale,
                            scale=(px, px), translate=(px / 2, px / 2))
    lyr.metadata["_palom_reader"] = reader


def _add_placeholder(viewer, name, shape, px):
    """A faint footprint-sized marker shown while the pyramid builds, so the user
    sees it's working (and doesn't re-drop). Replaced by the real layer when done."""
    H, W = int(shape[-2]), int(shape[-1])
    block = np.full((16, 16), 128, np.uint8)
    return viewer.add_image(block, name=f"{name} (building…)",
                            scale=(H * px / 16, W * px / 16),
                            translate=(px / 2, px / 2), colormap="gray",
                            contrast_limits=(0, 255), opacity=0.25,
                            blending="translucent")


def _cache_and_add(viewer, reader, px, name, *, channel, as_labels):
    level0 = reader.pyramid[0][channel]
    if as_labels and not np.issubdtype(level0.dtype, np.integer):
        level0 = level0.astype(np.int32)
    interp = cv2.INTER_NEAREST if as_labels else cv2.INTER_AREA
    compressor = (Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
                  if as_labels else "default")

    if name in _BUILDING:
        show_info(f"'{name}' is already building.")
        return
    _BUILDING.add(name)
    placeholder = _add_placeholder(viewer, name, level0.shape, px)
    show_info(f"Building '{name}' in the background…")

    def _cleanup():
        _BUILDING.discard(name)
        if placeholder in viewer.layers:
            viewer.layers.remove(placeholder)

    def _done(group):
        _cleanup()
        # palom's native level 0 on top, cached coarse levels below.
        coarse = [da.from_zarr(group[k]) for k in sorted(group.array_keys(), key=int)]
        levels = [level0, *coarse]
        if as_labels:
            lyr = viewer.add_labels(levels, name=name, multiscale=True,
                                    scale=(px, px), translate=(px / 2, px / 2))
        else:
            lyr = viewer.add_image(levels, name=name, multiscale=True,
                                   contrast_limits=(0, 5000),
                                   scale=(px, px), translate=(px / 2, px / 2))
        lyr.metadata["_palom_reader"] = reader   # keep level 0 readable
        show_info(f"'{name}' ready")

    def _on_err(e):
        _cleanup()
        print(f"DnD pyramid build error: {e}")
        show_info(f"Failed to build '{name}': {e}")

    worker = _build_pyramid_worker(level0, interp, compressor)
    worker.returned.connect(_done)
    worker.errored.connect(_on_err)
    worker.start()


# ── drop handling ──────────────────────────────────────────────────────────── #

def _handle_drop(viewer, path, default_px):
    if _layer_name(path) in _BUILDING:
        show_info(f"'{_layer_name(path)}' is still building — please wait.")
        return
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
        return   # reader released when the local goes out of scope

    dlg = _AddFileDialog(os.path.basename(path), (C, H, W), p0.dtype,
                         detected, score, prefill, ch_names)
    if not dlg.exec():
        return   # Cancel → reader released when the local goes out of scope

    typ, px, channels = dlg.layer_type(), dlg.pixel_size(), dlg.channels()
    name = _layer_name(path)
    # True stored pyramid vs palom's coarsen fallback — NOT len(reader.pyramid),
    # which is >1 in both cases.
    pyramidal = _is_pyramidal(path)

    if typ == "rgb":
        _add_rgb(viewer, reader, px, name)
    elif typ == "mask":
        if pyramidal:
            _add_mask_pyramidal(viewer, reader, px, name)   # real cheap levels
        else:
            # Non-pyramidal source: palom would synthesise coarse levels by
            # coarsening level 0 (mean — wrong for labels — and every coarse view
            # re-reads the full-res level 0, a multi-GB spike). Cache to a real
            # pyramid with strided nearest (also handles uint32 labels).
            _cache_and_add(viewer, reader, px, name, channel=0, as_labels=True)
    else:  # image
        if not pyramidal and len(channels) == 1 and max(H, W) > _CACHE_DIM_THRESHOLD:
            _cache_and_add(viewer, reader, px, name, channel=channels[0],
                           as_labels=False)
        else:
            _add_multichannel(viewer, reader, px, channels, ch_names)


class _DropFilter(QObject):
    def __init__(self, viewer, default_px_size):
        super().__init__()
        self._viewer = viewer
        self._default_px = default_px_size

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
                _handle_drop(self._viewer, p, self._default_px)
            return True
        return False


def install_drop_handler(viewer, default_px_size=None):
    """Intercept drops of palom-readable files and route them through a dialog.

    Keeps a reference to the filter on the QtViewer so it isn't garbage-collected.
    """
    qtv = viewer.window._qt_viewer
    qtv.setAcceptDrops(True)
    filt = _DropFilter(viewer, default_px_size)
    qtv.installEventFilter(filt)
    qtv._mask_tool_drop_filter = filt   # keep alive
    return filt
