#!/usr/bin/env python
"""
Mask-vs-intensity-image detection by foreground equality — self-contained.

A label mask is piecewise-constant: every pixel inside an object equals its
neighbours, only object boundaries differ. An intensity image has per-pixel
variation. The discriminator is the fraction of neighbouring *foreground* pixels
(both nonzero) that are identical:

    ~0.9 - 1.0   →  label mask      (regardless of label count / object size)
    ~0.0 - 0.05  →  intensity image

Background (zeros) is excluded, so a cropped FOV with large empty borders does not
false-positive.

Locating content cheaply: the metric needs full-res pixels (downsampling destroys
piecewise-constancy), but a whole-slide image is mostly glass. Rather than read the
entire image to find tissue, we sparse-sample tiles in a low-discrepancy order and
stop once enough foreground is seen — typically 1-2 tile reads. Only if sampling
can't find enough foreground (a tiny tissue fragment) do we fall back to a strided
coarse map that locates the densest tiles deterministically.

Works on:
  - a file path     (OME-TIFF / TIFF / SVS / VSI, read lazily via palom)
  - a palom reader  (anything with a `.pyramid` list of (C,H,W) dask arrays)
  - a 2-D or 3-D (C,H,W) numpy / dask array

This module is self-contained (no Qt / napari). `mask_tool.dnd` imports the
algorithm from here so there is a single source of truth.

CLI:
    python -m mask_tool.mask_detect IMAGE [IMAGE ...] [--channel N]

Import:
    from mask_tool.mask_detect import classify, foreground_equality
    label, score = classify("nucleus_mask.ome.tif")   # -> ("mask", 0.99)
    score = foreground_equality(my_2d_array)           # -> float or None

Dependencies: numpy, opencv-python (cv2), math; dask + palom only when the input
is a dask array or a file path.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except ImportError:            # only the coarse fallback needs cv2.resize
    cv2 = None

__all__ = [
    "classify", "foreground_equality", "equal_fg_pairs", "lds_order",
    "is_synthesized",
]

# ── tunables ──────────────────────────────────────────────────────────────── #
EQ_THRESHOLD  = 0.5        # equality above this → mask, below → image
MIN_PAIRS     = 100_000    # accumulate this many foreground pairs, then decide
MAX_TILES     = 6          # cap foreground tiles whose pairs we accumulate
PROBE_TILES   = 48         # max tiles the sparse sampler reads before falling back
TILE_MIN_FG   = 0.01       # skip probe tiles with foreground fraction below this
DEFAULT_TILE  = 2048       # tile edge when the array has no chunking (e.g. numpy)
COARSE_WINDOW = 1024       # full-res window edge for the coarse fallback
COARSE_SAMPLE = 2048       # strided coarse map ≈ this many px on the long edge


# ── core metric ───────────────────────────────────────────────────────────── #

def equal_fg_pairs(a: np.ndarray) -> tuple[int, int]:
    """(equal foreground pairs, total foreground pairs) over right+down neighbours.
    A pair counts only when *both* pixels are nonzero (background excluded)."""
    nz_r = (a[:, 1:] != 0) & (a[:, :-1] != 0)
    eq_r = nz_r & (a[:, 1:] == a[:, :-1])
    nz_d = (a[1:, :] != 0) & (a[:-1, :] != 0)
    eq_d = nz_d & (a[1:, :] == a[:-1, :])
    return int(eq_r.sum() + eq_d.sum()), int(nz_r.sum() + nz_d.sum())


def lds_order(n: int) -> list[int]:
    """Well-spread visiting order of ``range(n)``: a golden-ratio coprime stride
    (a low-discrepancy sequence), so probes scatter across the tile grid instead
    of clumping in one corner. Deterministic, no RNG."""
    if n <= 2:
        return list(range(n))
    k = max(1, round(n / 1.618033988749895))
    while k < n and math.gcd(k, n) != 1:
        k += 1
    if math.gcd(k, n) != 1:
        return list(range(n))
    return [(i * k) % n for i in range(n)]


def _tile_shape(plane) -> tuple[int, int]:
    cs = getattr(plane, "chunksize", None)        # dask arrays expose this
    if cs and len(cs) >= 2:
        return max(1, int(cs[-2])), max(1, int(cs[-1]))
    return DEFAULT_TILE, DEFAULT_TILE


def _sparse_equality(plane) -> float | None:
    """Fast path: probe chunk-aligned full-res tiles in a spread order, skip
    near-empty (background) tiles, and accumulate foreground pairs until
    ``MIN_PAIRS``. Reads only a handful of tiles — no whole-image read. Returns the
    equality fraction, or None if it can't find enough foreground within
    ``PROBE_TILES`` reads (caller falls back to the coarse map)."""
    H, W = int(plane.shape[-2]), int(plane.shape[-1])
    th, tw = _tile_shape(plane)
    n_ty, n_tx = max(1, -(-H // th)), max(1, -(-W // tw))   # ceil-div tile counts

    eq = tot = used = read = 0
    for idx in lds_order(n_ty * n_tx):
        if used >= MAX_TILES or tot >= MIN_PAIRS or read >= PROBE_TILES:
            break
        ty, tx = divmod(idx, n_tx)
        win = np.asarray(plane[ty * th:ty * th + th, tx * tw:tx * tw + tw])
        read += 1
        if win.size == 0 or (win != 0).mean() < TILE_MIN_FG:
            continue   # background-only tile — doesn't count toward the metric
        e, t = equal_fg_pairs(win)
        eq += e
        tot += t
        used += 1

    return eq / tot if tot >= MIN_PAIRS // 10 else None


def _coarse_equality(plane, coarse_level=None) -> float | None:
    """Deterministic fallback: build a coarse foreground map (a real downsampled
    level if given, else a strided sample of ``plane``), locate the densest tiles,
    and measure equality on full-res windows there."""
    if cv2 is None:
        raise RuntimeError("cv2 (opencv) is required for the coarse fallback")
    H, W = int(plane.shape[-2]), int(plane.shape[-1])
    if coarse_level is not None:
        coarse = np.asarray(coarse_level)
    else:
        s = max(1, max(H, W) // COARSE_SAMPLE)
        coarse = np.asarray(plane[::s, ::s])

    n_ty = max(1, H // COARSE_WINDOW)
    n_tx = max(1, W // COARSE_WINDOW)
    tile_fg = cv2.resize((coarse != 0).astype(np.float32), (n_tx, n_ty),
                         interpolation=cv2.INTER_AREA).ravel()
    order = np.argsort(tile_fg)[::-1]   # densest foreground first

    eq = tot = used = 0
    for idx in order:
        if used >= MAX_TILES or tot >= MIN_PAIRS:
            break
        if tile_fg[idx] <= 0:
            break
        ty, tx = divmod(int(idx), n_tx)
        y0, x0 = ty * COARSE_WINDOW, tx * COARSE_WINDOW
        win = np.asarray(plane[y0:y0 + COARSE_WINDOW, x0:x0 + COARSE_WINDOW])
        e, t = equal_fg_pairs(win)
        eq += e
        tot += t
        used += 1

    return eq / tot if tot >= MIN_PAIRS // 10 else None


def foreground_equality(plane, coarse_level=None) -> float | None:
    """Fraction of neighbouring foreground pixels that are identical for a 2-D
    array (label masks → ~1, intensity images → ~0). Sparse-samples first; reads
    the coarse map only if sampling can't find enough foreground. Returns None
    when content is too sparse to judge.

    ``coarse_level`` (optional): a real downsampled level to locate tissue in the
    fallback; if omitted a strided sample of ``plane`` is used."""
    score = _sparse_equality(plane)
    if score is not None:
        return score
    return _coarse_equality(plane, coarse_level)


# ── pyramid helpers / file-level classify ─────────────────────────────────── #

def is_synthesized(pyramid) -> bool:
    """True if palom synthesised the coarse levels by coarsening level 0 (vs.
    reading real stored levels) — the coarsest level's dask graph then shares
    layers with level 0's, whereas independently-read stored levels share none.
    Used to avoid forcing palom to coarsen the whole level 0 in the fallback.
    (Only meaningful for dask pyramids; returns False otherwise.)"""
    if len(pyramid) < 2:
        return False
    try:
        base = set(pyramid[0].__dask_graph__().layers)
        top = set(pyramid[-1].__dask_graph__().layers)
        return bool(base & top)
    except AttributeError:
        return False


def _as_pyramid(source) -> list:
    """Normalise ``source`` into a list of levels (level 0 first)."""
    if hasattr(source, "pyramid"):                    # palom reader
        return list(source.pyramid)
    if isinstance(source, str) or hasattr(source, "__fspath__"):
        import palom.reader as R
        path = str(source)
        low = path.lower()
        if low.endswith(".svs"):
            return list(R.SvsReader(path).pyramid)
        if low.endswith(".vsi"):
            return list(R.VsiReader(path).pyramid)
        return list(R.OmePyramidReader(path).pyramid)
    return [source]                                   # raw 2-D / 3-D array


def _coarse_for(pyramid, channel: int):
    """A real stored coarse level to locate tissue in the fallback, or None (then
    a strided sample of level 0 is used). Skipped for synthesised pyramids."""
    if len(pyramid) > 1 and not is_synthesized(pyramid):
        cl = pyramid[-1]
        return cl if cl.ndim == 2 else cl[channel]
    return None


def _classify_plane(plane, coarse_level=None) -> tuple[str, float | None]:
    score = foreground_equality(plane, coarse_level)
    if score is None:   # too sparse to judge → fall back to the dtype heuristic
        return ("mask" if np.issubdtype(plane.dtype, np.integer) else "image"), None
    return ("mask" if score > EQ_THRESHOLD else "image"), score


def classify(source, channel: int = 0) -> tuple[str, float | None]:
    """Classify ``source`` as ``("rgb" | "mask" | "image", equality_or_None)``.

    ``source`` may be a file path, a palom reader (``.pyramid``), or a 2-D / 3-D
    ``(C, H, W)`` numpy or dask array. For 3-D input: 3×uint8 → ``"rgb"``;
    otherwise ``channel`` is tested for mask-vs-image by foreground equality.
    """
    pyramid = _as_pyramid(source)
    p0 = pyramid[0]
    if p0.ndim == 2:
        return _classify_plane(p0, _coarse_for(pyramid, 0))

    C, dtype = int(p0.shape[0]), p0.dtype
    if C == 3 and dtype == np.uint8:
        return "rgb", None
    ch = channel if C > 1 else 0
    return _classify_plane(p0[ch], _coarse_for(pyramid, ch))


# ── CLI ───────────────────────────────────────────────────────────────────── #

def _main(argv=None) -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser(
        description="Classify image(s) as label mask vs intensity image.")
    ap.add_argument("images", nargs="+", help="image file path(s)")
    ap.add_argument("--channel", type=int, default=0,
                    help="channel index to test for multi-channel files (default 0)")
    args = ap.parse_args(argv)

    for path in args.images:
        t = time.perf_counter()
        try:
            label, score = classify(path, channel=args.channel)
        except Exception as e:                        # noqa: BLE001 — CLI surface
            print(f"ERROR  {path}: {e}")
            continue
        dt = time.perf_counter() - t
        s = f"{score:.4f}" if score is not None else "  n/a "
        print(f"{label:5s}  equality={s}  ({dt:6.2f}s)  {path}")


if __name__ == "__main__":
    _main()
