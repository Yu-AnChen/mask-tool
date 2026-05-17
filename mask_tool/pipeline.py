"""
Per-channel mask building and multi-mask combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

COMBINE_OPS = ("AND", "OR", "AND NOT", "OR NOT", "XOR")


# ── morphological helpers (OpenCV-based, no skimage dependency) ─────────────


def remove_small_objects(
    ar: np.ndarray, min_size: int = 64, connectivity: int = 1
) -> np.ndarray:
    binary = (ar > 0).astype(np.uint8) * 255
    cv2_conn = 4 if connectivity == 1 else 8
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=cv2_conn)
    keep = np.nonzero(stats[1:, cv2.CC_STAT_AREA] >= min_size)[0] + 1
    out = np.isin(labels, keep)
    if ar.dtype == bool:
        return out
    return out.astype(ar.dtype)


def remove_small_holes(
    ar: np.ndarray, area_threshold: int = 64, connectivity: int = 1
) -> np.ndarray:
    binary = (ar > 0).astype(np.uint8) * 255
    inverted = cv2.bitwise_not(binary)
    cv2_conn = 4 if connectivity == 1 else 8
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=cv2_conn)

    border_labels = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ]))
    areas = stats[1:, cv2.CC_STAT_AREA]
    fill = np.nonzero((areas <= area_threshold) & ~np.isin(np.arange(1, n), border_labels))[0] + 1
    out = binary.copy()
    out[np.isin(labels, fill)] = 255
    if ar.dtype == bool:
        return out > 0
    return (out > 0).astype(ar.dtype)


# ── parameter dataclass ──────────────────────────────────────────────────────


@dataclass
class CombineParams:
    steps: list[dict] = field(default_factory=list)  # [{"layer": str}, {"op": str, "layer": str}, ...]
    hole_threshold: int = 10
    obj_threshold: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


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
