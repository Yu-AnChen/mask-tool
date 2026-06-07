"""
Shared multiscale-pyramid builder for caching large 2-D arrays to a zarr group.

Used by:
  - rolling_ball.subtract_background  (INTER_AREA, intensity data)
  - dnd                               (INTER_AREA for images, INTER_NEAREST for
                                        label masks — averaging label IDs is
                                        meaningless, so masks must use nearest)

A pyramid is a zarr group with string-keyed levels "0", "1", … : level 0 is the
full-resolution input; each successive level is a `factor`× downsample. Levels
are written one at a time, each read back from disk before producing the next,
so peak memory stays bounded and an expensive level-0 graph is computed once.

Exactness: each level is downsampled per-dask-block. As long as the input chunk
edges are multiples of `factor`, a per-block resize is identical to a whole-image
resize (no seams). `_downsample` enforces this by rechunking input to
`out_chunk * factor` before the resize.
"""

from __future__ import annotations

import cv2
import numpy as np
import zarr
import dask.array as da


def _downsample(x: "da.Array", factor: int, interpolation: int,
                out_chunk: int) -> "da.Array":
    """Lazy `factor`× downsample of a 2-D dask array.

    Input is rechunked to `out_chunk * factor` so every block emits a clean
    `out_chunk`-pixel output chunk; because chunk edges stay a multiple of
    `factor`, the per-block resize equals a whole-image resize (no seams).
    """
    f = factor
    x = x.rechunk((out_chunk * f, out_chunk * f))
    out_chunks = tuple(tuple(-(-c // f) for c in ax) for ax in x.chunks)

    if interpolation == cv2.INTER_NEAREST:
        # Strided slicing: dtype-agnostic (cv2.resize rejects uint32/uint64,
        # common for instance-label masks) and exactly nearest. len(range(0,c,f))
        # == ceil(c/f), so output chunks match out_chunks.
        def _resize(block: np.ndarray) -> np.ndarray:
            return block[::f, ::f]
    else:
        def _resize(block: np.ndarray) -> np.ndarray:
            h, w = block.shape
            return cv2.resize(block, (-(-w // f), -(-h // f)), interpolation=interpolation)

    return x.map_blocks(_resize, dtype=x.dtype, chunks=out_chunks)


def write_pyramid_group(
    level0: "da.Array",
    out_path: "str | None",
    *,
    chunk0: int = 2048,
    chunk_lo: int = 1024,
    n_levels: int = 3,
    factor: int = 4,
    interpolation: int = cv2.INTER_AREA,
    dask_workers: int | None = None,
    compressor: object = "default",
    store_level0: bool = True,
) -> "zarr.hierarchy.Group":
    """Write `level0` (a 2-D dask array) and its downsamples to a zarr group as
    multiscale levels.

    Parameters
    ----------
    level0        : full-resolution 2-D dask array
    out_path      : path for the output zarr group, or None for an in-memory
                    (RAM) zarr group — chunks stay zstd/Blosc-compressed in RAM,
                    so tiny coarse levels need no disk cache
    chunk0        : chunk edge for level 0
    chunk_lo      : chunk edge for levels ≥ 1
    n_levels      : total number of levels including level 0
    factor        : downsample factor between levels
    interpolation : cv2 interpolation (INTER_AREA for intensity, INTER_NEAREST
                    for label masks)
    dask_workers  : threads for da.store; None → dask default
    compressor    : numcodecs codec, None (uncompressed), or "default" (zarr's
                    default Blosc); pass a zstd Blosc for label masks
    store_level0  : if False, don't write level "0" — only the coarse levels
                    "1".."{n_levels-1}" (level 1 is downsampled straight from the
                    in-memory `level0`). Use when the caller already has a cheap
                    full-res source (e.g. a reader's native level 0) and only needs
                    cheap coarse levels; avoids re-reading/re-compressing level 0.

    Returns
    -------
    The zarr group. Levels "0".."{n_levels-1}" if `store_level0`, else
    "1".."{n_levels-1}".
    """
    group = (zarr.group(store=zarr.MemoryStore()) if out_path is None
             else zarr.open_group(out_path, mode="w"))
    dtype = level0.dtype
    if store_level0:
        level, first = level0, 0
    else:
        level, first = _downsample(level0, factor, interpolation, chunk_lo), 1
    for i in range(first, n_levels):
        cs = chunk0 if i == 0 else chunk_lo
        level = level.rechunk((cs, cs))
        out = group.zeros(str(i), shape=level.shape, chunks=(cs, cs),
                          dtype=dtype, compressor=compressor)
        # callbacks=[] keeps this store out of dask's *global* callbacks. napari
        # registers a process-global opportunistic `dask.cache.Cache` while it
        # slices dask layers; that cache's callbacks aren't safe across two
        # concurrent dask schedulers, so a background build racing a foreground
        # slice corrupts its `starttimes` → KeyError. Passing a non-None list
        # makes dask use no callbacks here and skip the global-callback swap,
        # isolating this store from (and not interfering with) napari's reads.
        da.store(level, out, scheduler="threads", num_workers=dask_workers,
                 callbacks=[])
        if i < n_levels - 1:
            level = _downsample(da.from_zarr(out), factor, interpolation, chunk_lo)
    return group
