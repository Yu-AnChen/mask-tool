import pathlib

import cv2
import geojson
import napari
import numpy as np
import ome_types
import palom
import shapely
import shapely.affinity
import tifffile

geojson.geometry.DEFAULT_PRECISION = 2


def remove_small_objects(
    ar: np.ndarray, min_size: int = 64, connectivity: int = 1
) -> np.ndarray:
    """
    Remove connected components smaller than the specified size.

    Replicates skimage.morphology.remove_small_objects using OpenCV.

    Parameters
    ----------
    ar : np.ndarray
        Boolean or integer binary array. Nonzero values are treated as foreground.
    min_size : int
        Minimum number of pixels in a connected component to keep.
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity.

    Returns
    -------
    np.ndarray
        Array with small objects removed, same dtype as input.
    """
    if ar.dtype != np.uint8:
        binary = (ar > 0).astype(np.uint8) * 255
    else:
        binary = (ar > 0).astype(np.uint8) * 255

    cv2_connectivity = 4 if connectivity == 1 else 8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=cv2_connectivity
    )

    # stats[:, cv2.CC_STAT_AREA] gives pixel count per label (label 0 = background)
    out = np.zeros_like(binary)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            out[labels == label] = 255

    # Return in original dtype: bool stays bool, others stay as-is
    if ar.dtype == bool:
        return out > 0
    return (out > 0).astype(ar.dtype)


def remove_small_holes(
    ar: np.ndarray, area_threshold: int = 64, connectivity: int = 1
) -> np.ndarray:
    """
    Remove contiguous holes smaller than the specified size.

    Replicates skimage.morphology.remove_small_holes using OpenCV.

    Parameters
    ----------
    ar : np.ndarray
        Boolean or integer binary array. Nonzero values are treated as foreground.
    area_threshold : int
        Holes with pixel count <= this value are filled.
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity.

    Returns
    -------
    np.ndarray
        Array with small holes filled, same dtype as input.
    """
    if ar.dtype != np.uint8:
        binary = (ar > 0).astype(np.uint8) * 255
    else:
        binary = (ar > 0).astype(np.uint8) * 255

    # Holes are connected components of the *background* (inverted image)
    inverted = cv2.bitwise_not(binary)

    cv2_connectivity = 4 if connectivity == 1 else 8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=cv2_connectivity
    )

    # Label 0 in the inverted image corresponds to the original foreground —
    # skip it. All other labels are background regions (holes or outer border).
    # We need to identify which label touches the image border (true background)
    # and leave it alone; only fill interior holes.
    h, w = binary.shape

    border_labels = set()
    border_labels.update(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())

    out = binary.copy()
    for label in range(1, num_labels):
        if label in border_labels:
            continue  # True background — do not fill
        if stats[label, cv2.CC_STAT_AREA] <= area_threshold:
            out[labels == label] = 255  # Fill the hole

    if ar.dtype == bool:
        return out > 0
    return (out > 0).astype(ar.dtype)


def mask_to_polygon(mask):
    # should also test/compare with the rasterio.features.shapes
    # https://gist.github.com/petebankhead/77782fd6d684e18efb2447980fdfbb90
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    # [next, previous, first child, parent]
    hierarchy = hierarchy.squeeze()

    polygons = []
    base_holes = []
    for ii, (ee, hh) in enumerate(zip(contours, hierarchy)):
        if hh[3] != -1:
            continue
        ee = np.atleast_2d(np.squeeze(ee))
        # FIXME: find out an edge case for this
        if np.any(ee < 0) or np.any(ee >= mask.shape[::-1]):
            continue
        # "exterior" of the masks
        if mask[ee.T[1], ee.T[0]].sum() == 0:
            continue
        # insufficient number of vertices
        if len(ee) < 3:
            continue
        # holes that are connected to the background
        if shapely.LinearRing(ee).is_ccw:
            base_holes.append(shapely.Polygon(shell=ee))
            continue
        interiors = []
        if hh[2] != -1:
            hole_idxs = np.where(hierarchy[:, 3] == ii)[0]
            for hole_idx in hole_idxs:
                hole = np.atleast_2d(np.squeeze(contours[hole_idx]))
                if len(hole) < 3:
                    continue
                interiors.append(hole)
        polygons.append(shapely.Polygon(shell=ee, holes=interiors))
    multi_polygon = shapely.MultiPolygon(polygons) - shapely.MultiPolygon(base_holes)
    return shapely.MultiPolygon(
        [pp for pp in shapely.get_parts(multi_polygon) if pp.geom_type == "Polygon"]
    )


out_dir = r"\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_NucleAI_samples\QuPath-ROI\ROI-tumor-v3"


files = r"""
\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S324_or19_A107_full_001\Q127_S324_012653\Q127_S324_or19_A107_full_001_Q127_S324_012653.ome.tiff
\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S319_or19_A107_full_001\Q127_S319_012652\Q127_S319_or19_A107_full_001_Q127_S319_012652.ome.tiff
\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S314_or19_A107_full_001\Q127_S314_012651\Q127_S314_or19_A107_full_001_Q127_S314_012651.ome.tiff
\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S307_or19_A107_full_001\Q127_S307_012650\Q127_S307_or19_A107_full_001_Q127_S307_012650.ome.tiff
""".strip().split("\n")

for ff in files:
    file_path = ff

    print(f"Read image {file_path}")
    reader = palom.reader.OmePyramidReader(file_path)
    px_size_img = reader.pixel_size

    px_size_mask = 10
    factor_mask = px_size_img / px_size_mask

    cutoff_ck = 400
    cutoff_ecad = 650

    img_ntform = dict(
        scale=(px_size_img, px_size_img), translate=px_size_img * np.array([0.5, 0.5])
    )
    mask_ntform = dict(
        scale=(px_size_mask, px_size_mask),
        translate=px_size_mask * np.array([0.5, 0.5]),
    )
    channel_names = [
        cc.name for cc in ome_types.from_tiff(file_path).images[0].pixels.channels
    ]

    print(f"Make tumor mask with CK ({cutoff_ck}) | Ecad ({cutoff_ecad})")
    ecad = cv2.resize(
        tifffile.imread(reader.path, key=10),
        dsize=None,
        fx=factor_mask,
        fy=factor_mask,
        interpolation=cv2.INTER_AREA,
    )
    ck = cv2.resize(
        tifffile.imread(reader.path, key=18),
        dsize=None,
        fx=factor_mask,
        fy=factor_mask,
        interpolation=cv2.INTER_AREA,
    )
    mask = (cv2.GaussianBlur(ck, ksize=None, sigmaX=1, sigmaY=1) > cutoff_ck) | (
        cv2.GaussianBlur(ecad, ksize=None, sigmaX=1, sigmaY=1) > cutoff_ecad
    )
    mask_clean = remove_small_objects(remove_small_holes(mask, 10), 10, connectivity=2)

    geom = mask_to_polygon(mask_clean.astype("uint8"))
    geom = shapely.affinity.scale(
        geom, xfact=1 / factor_mask, yfact=1 / factor_mask, origin=(0, 0)
    )

    geojson_obj = geojson.Feature(
        geometry=geom,
        properties=dict(name=f"Tumor: CK > {cutoff_ck} | ECAD > {cutoff_ecad}"),
    )
    out_path = pathlib.Path(out_dir) / (reader.path.name.split(".")[0] + ".geojson")
    with open(out_path, "w") as f:
        geojson.dump(geojson_obj, f)
# ---------------------------------------------------------------------------- #
#                                  napari viz                                  #
# ---------------------------------------------------------------------------- #
v = napari.Viewer()
v.add_image(
    [pp[[0, 10, 18]] for pp in reader.pyramid],
    channel_axis=0,
    name=np.array(channel_names)[[0, 10, 18]],
    visible=False,
    contrast_limits=(0, 5000),
    **img_ntform,
)
v.add_image(ecad, **mask_ntform)
v.add_image(ck, blending="additive", **mask_ntform)
v.add_labels(mask, **mask_ntform)

print("Clean up mask")
v.add_labels(mask_clean, **mask_ntform)
