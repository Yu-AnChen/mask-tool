"""
Per-channel mask building and multi-mask combination.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
import dask.array as da
import zarr

from .resize import lazy_resize
from .rolling_ball import subtract_background


COMBINE_OPS = ("AND", "OR", "AND NOT", "OR NOT", "XOR")


# ── morphological helpers (OpenCV-based, no skimage dependency) ─────────────


def remove_small_objects(
    ar: np.ndarray, min_size: int = 64, connectivity: int = 1
) -> np.ndarray:
    binary = (ar > 0).astype(np.uint8) * 255
    cv2_conn = 4 if connectivity == 1 else 8
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=cv2_conn)
    out = np.zeros_like(binary)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_size:
            out[labels == lbl] = 255
    if ar.dtype == bool:
        return out > 0
    return (out > 0).astype(ar.dtype)


def remove_small_holes(
    ar: np.ndarray, area_threshold: int = 64, connectivity: int = 1
) -> np.ndarray:
    binary = (ar > 0).astype(np.uint8) * 255
    inverted = cv2.bitwise_not(binary)
    cv2_conn = 4 if connectivity == 1 else 8
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=cv2_conn)

    border = set()
    border.update(labels[0, :].tolist())
    border.update(labels[-1, :].tolist())
    border.update(labels[:, 0].tolist())
    border.update(labels[:, -1].tolist())

    out = binary.copy()
    for lbl in range(1, n):
        if lbl not in border and stats[lbl, cv2.CC_STAT_AREA] <= area_threshold:
            out[labels == lbl] = 255
    if ar.dtype == bool:
        return out > 0
    return (out > 0).astype(ar.dtype)


# ── parameter dataclass ──────────────────────────────────────────────────────


@dataclass
class ChannelMaskParams:
    layer_name: str
    channel_idx: int
    px_size_src: float
    target_px_size: float
    bg_subtract: bool = False
    rolling_ball_radius: float = 50.0
    gaussian_sigma: float = 1.0
    threshold: float = 400.0
    hole_threshold: int = 10
    obj_threshold: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CombineParams:
    steps: list[dict] = field(default_factory=list)  # [{"layer": str}, {"op": str, "layer": str}, ...]
    hole_threshold: int = 10
    obj_threshold: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


# ── channel mask ─────────────────────────────────────────────────────────────


def build_channel_mask(
    src: da.Array | zarr.Array | np.ndarray,
    px_size_src: float,
    target_px_size: float,
    *,
    bg_subtract: bool = False,
    rolling_ball_radius: float = 50.0,
    gaussian_sigma: float = 1.0,
    threshold: float = 400.0,
    hole_threshold: int = 10,
    obj_threshold: int = 10,
    chunk_size: int = 2048,
    num_workers: int | None = None,
    existing_cache: zarr.Array | None = None,
) -> tuple[zarr.Array, np.ndarray]:
    """
    Downsample one channel, optionally subtract background, then threshold.

    Returns
    -------
    cache : zarr.Array
        The downsampled (and optionally BG-subtracted) float32 image.
        Kept alive by the caller for re-use across threshold adjustments.
    mask : np.ndarray (bool)
        Binary mask after smoothing, thresholding, and morphological cleanup.
    """
    scale = px_size_src / target_px_size
    n_workers = num_workers or os.cpu_count() or 1

    if existing_cache is None:
        if isinstance(src, zarr.Array):
            x = da.from_zarr(src, chunks=chunk_size)
        elif isinstance(src, np.ndarray):
            x = da.from_array(src, chunks=chunk_size)
        else:
            x = src

        x_small = lazy_resize(x, scale=scale, chunk_size=chunk_size)

        cache = zarr.zeros(x_small.shape, chunks=min(512, max(x_small.shape)), dtype=np.float32)
        da.store(x_small, cache, scheduler="threads", num_workers=n_workers)

        if bg_subtract:
            cache = subtract_background(
                cache,
                radius=rolling_ball_radius,
                num_workers=n_workers,
            )
    else:
        cache = existing_cache

    img = np.asarray(cache[:])
    if gaussian_sigma > 0:
        img = cv2.GaussianBlur(
            img, ksize=None, sigmaX=gaussian_sigma, sigmaY=gaussian_sigma
        )

    mask = img > threshold
    mask = remove_small_holes(mask, hole_threshold, connectivity=2)
    mask = remove_small_objects(mask, obj_threshold, connectivity=2)

    return cache, mask


# ── mask combination ─────────────────────────────────────────────────────────


def _apply_op(acc: np.ndarray, op: str, b: np.ndarray) -> np.ndarray:
    if op == "AND":
        return acc & b
    if op == "OR":
        return acc | b
    if op == "AND NOT":
        return acc & ~b
    if op == "OR NOT":
        return acc | ~b
    if op == "XOR":
        return acc ^ b
    raise ValueError(f"Unknown op: {op!r}")


def combine_masks(
    masks: list[np.ndarray],
    ops: list[str],
    *,
    hole_threshold: int = 0,
    obj_threshold: int = 0,
) -> np.ndarray:
    """
    Combine 2+ boolean masks with left-to-right logical operations.

    Parameters
    ----------
    masks : list of bool arrays, all same shape
    ops   : list of op strings, len == len(masks) - 1
            each op in COMBINE_OPS
    """
    if len(masks) < 2:
        raise ValueError("Need at least 2 masks to combine")
    if len(ops) != len(masks) - 1:
        raise ValueError("len(ops) must equal len(masks) - 1")

    result = masks[0].astype(bool)
    for op, m in zip(ops, masks[1:]):
        result = _apply_op(result, op, m.astype(bool))

    if hole_threshold > 0:
        result = remove_small_holes(result, hole_threshold, connectivity=2)
    if obj_threshold > 0:
        result = remove_small_objects(result, obj_threshold, connectivity=2)

    return result
