"""
Lazy anti-aliasing downscale for large 2-D dask arrays.

Uses dask.array.overlap so every output pixel has full INTER_AREA kernel
support at chunk boundaries — no aliasing seams for non-integer scale factors.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import dask.array as da
from dask.array.overlap import overlap as _da_overlap


def _resize_block(
    block: np.ndarray,
    *,
    scale: float,
    depth_out: int,
    block_info=None,
) -> np.ndarray:
    resized = cv2.resize(
        block.astype(np.float32),
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    d = depth_out
    trimmed = resized[d:-d, d:-d]

    # enforce exact declared chunk size — INTER_AREA rounding can give ±1 pixel
    if block_info is not None:
        th, tw = block_info[None]["chunk-shape"]
        ah, aw = trimmed.shape
        if ah < th or aw < tw:
            trimmed = np.pad(
                trimmed,
                ((0, max(0, th - ah)), (0, max(0, tw - aw))),
                mode="edge",
            )
        trimmed = trimmed[:th, :tw]
    return trimmed


def lazy_resize(
    x: da.Array,
    scale: float,
    chunk_size: int = 2048,
) -> da.Array:
    """
    Return a lazy dask array that is `x` downsampled by `scale` using INTER_AREA.

    Parameters
    ----------
    x          : 2-D dask array (any numeric dtype)
    scale      : target_px_size / source_px_size  (< 1 for downscaling)
    chunk_size : source-space chunk edge in pixels
    """
    if abs(scale - 1.0) < 1e-9:
        return x.astype(np.float32)

    if x.ndim != 2:
        raise ValueError(f"lazy_resize expects a 2-D array, got shape {x.shape}")

    depth_in = math.ceil(1.0 / scale)
    depth_out = max(1, round(depth_in * scale))

    x = x.rechunk({0: chunk_size, 1: chunk_size})

    out_chunks = tuple(
        tuple(round(c * scale) for c in dim) for dim in x.chunks
    )

    x_ghost = _da_overlap(
        x.astype(np.float32),
        depth={0: depth_in, 1: depth_in},
        boundary="reflect",
    )

    def _block(block, block_info=None):
        return _resize_block(block, scale=scale, depth_out=depth_out, block_info=block_info)

    return da.map_blocks(_block, x_ghost, chunks=out_chunks, dtype=np.float32)
