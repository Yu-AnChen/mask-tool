"""
Export final mask to GeoJSON, Zarr, or OME-TIFF, and write parameter logs.
"""

from __future__ import annotations

import json
import pathlib
import zipfile
from datetime import datetime

import cv2
import geojson
import numpy as np
import shapely
import shapely.affinity
import zarr

geojson.geometry.DEFAULT_PRECISION = 2


# ── mask → polygon (from polygon_util.py) ────────────────────────────────────


def mask_to_polygon(mask: np.ndarray) -> shapely.MultiPolygon:
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    hierarchy = hierarchy.squeeze()

    polygons = []
    base_holes = []
    for ii, (ee, hh) in enumerate(zip(contours, hierarchy)):
        if hh[3] != -1:
            continue
        ee = np.atleast_2d(np.squeeze(ee))
        if np.any(ee < 0) or np.any(ee >= mask.shape[::-1]):
            continue
        if mask[ee.T[1], ee.T[0]].sum() == 0:
            continue
        if len(ee) < 3:
            continue
        if shapely.LinearRing(ee).is_ccw:
            base_holes.append(shapely.Polygon(shell=ee))
            continue
        interiors = []
        if hh[2] != -1:
            for hole_idx in np.where(hierarchy[:, 3] == ii)[0]:
                hole = np.atleast_2d(np.squeeze(contours[hole_idx]))
                if len(hole) >= 3:
                    interiors.append(hole)
        polygons.append(shapely.Polygon(shell=ee, holes=interiors))

    multi = shapely.MultiPolygon(polygons) - shapely.MultiPolygon(base_holes)
    return shapely.MultiPolygon(
        [p for p in shapely.get_parts(multi) if p.geom_type == "Polygon"]
    )


# ── exporters ────────────────────────────────────────────────────────────────


def export_geojson(
    mask: np.ndarray,
    pixel_size: float,
    out_path: str | pathlib.Path,
    properties: dict | None = None,
    compress: bool = False,
) -> pathlib.Path:
    """
    Convert binary mask to a (Multi)Polygon and write as GeoJSON.

    Coordinates are in full-resolution pixel space (scaled by pixel_size).
    Compatible with QuPath annotation import.
    """
    out_path = pathlib.Path(out_path)
    geom = mask_to_polygon(mask.astype(np.uint8))
    # scale from mask-pixel space back to full-res pixel space
    geom = shapely.affinity.scale(geom, xfact=pixel_size, yfact=pixel_size, origin=(0, 0))

    feature = geojson.Feature(
        geometry=geom,
        properties=properties or {},
    )

    if compress:
        zip_path = out_path.with_suffix(".geojson.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.geojson", geojson.dumps(feature))
        return zip_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        geojson.dump(feature, f)
    return out_path


def export_zarr(
    mask: np.ndarray,
    out_path: str | pathlib.Path,
    pixel_size: float | None = None,
) -> zarr.Array:
    """Write binary mask as a Zarr array. Stores pixel_size in attrs if given."""
    out_path = str(out_path)
    arr = zarr.open_array(
        out_path,
        mode="w",
        shape=mask.shape,
        dtype=bool,
        chunks=min(512, max(mask.shape)),
    )
    arr[:] = mask
    if pixel_size is not None:
        arr.attrs["pixel_size_um"] = pixel_size
    return arr


def export_tiff(
    mask: np.ndarray,
    out_path: str | pathlib.Path,
    pixel_size: float = 1.0,
) -> pathlib.Path:
    """Write binary mask as a pyramidal OME-TIFF (palom, is_mask=True)."""
    from palom.pyramid import write_pyramid
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_pyramid(
        [mask.astype(np.uint8) * 255],
        out_path,
        pixel_size=pixel_size,
        is_mask=True,
        compression="zlib",
    )
    return out_path


# ── parameter log ─────────────────────────────────────────────────────────────


def save_params(params: dict, out_path: str | pathlib.Path) -> pathlib.Path:
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), **params}
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return out_path
