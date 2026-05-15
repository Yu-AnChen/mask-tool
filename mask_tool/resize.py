"""
Lazy anti-aliasing downscale for large 2-D dask arrays.

Chunks are aligned to INTER_AREA kernel boundaries (multiples of the scale
denominator) so chunk-independent resize is pixel-identical to a full-image
resize — no seams at chunk boundaries.
"""

from __future__ import annotations

import math
from fractions import Fraction

import cv2
import numpy as np
import dask.array as da


def _resize_block(block: np.ndarray, *, scale: float, block_info=None) -> np.ndarray:
    resized = cv2.resize(
        block,
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    if block_info is not None:
        th, tw = block_info[None]["chunk-shape"]
        ah, aw = resized.shape
        if ah < th or aw < tw:
            resized = np.pad(
                resized,
                ((0, max(0, th - ah)), (0, max(0, tw - aw))),
                mode="edge",
            )
        resized = resized[:th, :tw]
    return resized


def _make_chunks(dim_size: int, aligned: int, min_chunk: int) -> tuple[int, ...]:
    """
    Chunk `dim_size` into pieces of `aligned` pixels, ensuring the last
    (remainder) chunk is at least `min_chunk` — if not, absorb it into the
    previous chunk so cv2.resize always produces ≥1 output pixel per chunk.
    """
    if dim_size <= aligned:
        return (dim_size,)
    full, rem = divmod(dim_size, aligned)
    if rem == 0:
        return (aligned,) * full
    if rem >= min_chunk:
        return (aligned,) * full + (rem,)
    # remainder too small to produce any output — absorb into previous chunk
    return (aligned,) * (full - 1) + (aligned + rem,)


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
    chunk_size : source-space chunk edge in pixels (rounded up to scale denominator)
    """
    if abs(scale - 1.0) < 1e-9:
        return x

    if x.ndim != 2:
        raise ValueError(f"lazy_resize expects a 2-D array, got shape {x.shape}")

    # Align chunk size to the denominator of scale (in lowest terms) so that
    # INTER_AREA kernel boundaries fall exactly on chunk edges — no partial
    # pixels straddle chunk boundaries, giving seam-free results.
    q = Fraction(scale).limit_denominator(1000).denominator
    aligned = ((chunk_size + q - 1) // q) * q  # round up to next multiple of q

    # cv2.resize errors when output dimension rounds to 0; guard the last
    # (potentially small remainder) chunk: need input >= ceil(0.5 / scale)
    min_chunk = math.ceil(0.5 / scale)

    chunks_0 = _make_chunks(x.shape[0], aligned, min_chunk)
    chunks_1 = _make_chunks(x.shape[1], aligned, min_chunk)
    x = x.rechunk({0: chunks_0, 1: chunks_1})

    def _cumulative_chunks(in_chunks: tuple[int, ...]) -> tuple[int, ...]:
        """
        Compute output chunk sizes via cumulative rounding so they sum to
        exactly round(total_input * scale), avoiding per-chunk drift.
        """
        out, pos = [], 0
        for c in in_chunks:
            out.append(round((pos + c) * scale) - round(pos * scale))
            pos += c
        return tuple(out)

    out_chunks = tuple(_cumulative_chunks(dim) for dim in x.chunks)
    src_dtype = x.dtype

    def _block(block, block_info=None):
        return _resize_block(block, scale=scale, block_info=block_info)

    return da.map_blocks(_block, x, chunks=out_chunks, dtype=src_dtype)
