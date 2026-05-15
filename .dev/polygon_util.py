import shapely
import cv2
import numpy as np
import skimage.segmentation
import zarr
import tqdm
import skimage.transform
import warnings


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


def _polygon_to_mask(polygon: shapely.Polygon, fill_value=1, return_offset=True):
    minx, miny = np.floor(polygon.bounds[0:2]).astype("int") - 1
    maxx, maxy = np.ceil(polygon.bounds[2:4]).astype("int") + 1

    img = np.zeros((maxy - miny, maxx - minx), dtype="uint8")
    polygons = shapely.get_parts(polygon)
    for pp in polygons:
        contours = [np.round(rr.coords).astype("int") for rr in shapely.get_rings(pp)]
        cv2.fillPoly(
            img,
            contours,
            fill_value,
            offset=[-minx, -miny],
        )
    if not return_offset:
        return img
    return img, (miny, minx)


def polygons_to_mask(
    polygon: shapely.Polygon | shapely.MultiPolygon, binarize: bool = False
):
    minx, miny = np.floor(polygon.bounds[0:2]).astype("int") - 1
    maxx, maxy = np.ceil(polygon.bounds[2:4]).astype("int") + 1

    num_polygons = shapely.get_num_geometries(polygon)
    if num_polygons == 0:
        raise ValueError("No polygons found in the input.")

    dtype = "uint8"
    if not binarize:
        img_dtypes = ["uint8", "uint16", "int32"]
        max_idxs = [np.iinfo(dd).max for dd in img_dtypes]
        try:
            dtype = img_dtypes[np.argwhere(max_idxs > num_polygons).min()]
        except ValueError:
            raise ValueError(
                f"Number of polygons {num_polygons} exceeds the maximum supported "
                f"number of polygons {max(max_idxs)} for the given data types."
            )

    img = zarr.zeros((maxy - miny, maxx - minx), dtype=dtype)
    func = tqdm.tqdm if shapely.get_num_geometries(polygon) > 1 else np.asarray
    for ii, pp in enumerate(func(shapely.get_parts(polygon))):
        ii += 1
        if binarize:
            ii = 1
        mask, (row_s, col_s) = _polygon_to_mask(pp, return_offset=True)
        row_s -= miny
        col_s -= minx
        rr, cc = np.nonzero(mask)
        rr += row_s
        cc += col_s
        img.vindex[rr, cc] = ii
    return img, (miny, minx)


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


def expand_polygon(
    polygons: shapely.Polygon | shapely.MultiPolygon,
    expand_size: float,
):
    assert expand_size > 0, "expand_size must be positive"

    mask, offset_yx = polygons_to_mask(polygons.buffer(expand_size), binarize=True)

    dtype = "int32"
    mask = zarr.zeros(mask.shape, dtype=dtype)

    ori_mask, ori_offset_yx = polygons_to_mask(polygons)

    rs, cs = np.subtract(ori_offset_yx, offset_yx)
    h, w = ori_mask.shape
    # NOTE: may be out of range when expand_size is small
    mask[rs : rs + h, cs : cs + w] = ori_mask

    mask = skimage.segmentation.expand_labels(mask[:], distance=expand_size)
    expanded_polygons = mask_to_polygon(np.asarray(mask))
    expanded_polygons = shapely.transform(
        expanded_polygons,
        skimage.transform.AffineTransform(translation=offset_yx[::-1]),
    )
    expanded_polygons = align_expanded_polygon_to_original(expanded_polygons, polygons)
    return expanded_polygons


def expand_polygon_deprecated(
    polygons: shapely.Polygon | shapely.MultiPolygon,
    expand_size: float,
    _batch_size: int = 100_000,
):
    # deprecated in favor of using skimage.segmentation.expand_labels
    assert expand_size > 0, "expand_size must be positive"

    mask, (miny, minx) = polygons_to_mask(polygons.buffer(expand_size), binarize=True)

    dtype = "int32"
    mask = zarr.zeros(mask.shape, dtype=dtype)
    mask_conflict = zarr.zeros(mask.shape, dtype="uint8")

    buffered_polygons = [pp.buffer(expand_size) for pp in shapely.get_parts(polygons)]

    for ii, bp in enumerate(tqdm.tqdm(buffered_polygons, desc="Expanding polygons")):
        bmask, (bminy, bminx) = polygons_to_mask(bp, binarize=True)
        brr, bcc = np.nonzero(bmask)

        mask.vindex[brr + bminy - miny, bcc + bminx - minx] = ii + 1
        mask_conflict.vindex[brr + bminy - miny, bcc + bminx - minx] += 1

    mask_ori, (ominy, ominx) = polygons_to_mask(polygons, binarize=False)
    orr, occ = np.nonzero(mask_ori)

    # skip pre-expansion polygon pixels
    mask_conflict.vindex[orr + ominy - miny, occ + ominx - minx] = 0
    mask.vindex[orr + ominy - miny, occ + ominx - minx] = mask_ori[orr, occ]

    rows, cols = np.where(mask_conflict[:] > 1)
    del mask_conflict, buffered_polygons, mask_ori, orr, occ

    tformed_polygons = shapely.transform(
        polygons,
        skimage.transform.AffineTransform(translation=(minx, miny)).inverse,
    )
    shapely.prepare(tformed_polygons)
    tree = shapely.STRtree(tformed_polygons.geoms)

    _n_batch = len(rows) // _batch_size + 1
    for ii in tqdm.tqdm(range(_n_batch), desc="Querying nearest"):
        _start = ii * _batch_size
        _end = min((ii + 1) * _batch_size, len(rows))

        points = shapely.points(cols[_start:_end], rows[_start:_end])
        qresult = tree.query_nearest(
            points,
            max_distance=expand_size,
            return_distance=False,
            exclusive=False,
            all_matches=False,
        )

        mask.vindex[rows[_start:_end][qresult[0]], cols[_start:_end][qresult[0]]] = (
            qresult[1] + 1
        )

    expanded_polygons = mask_to_polygon(np.asarray(mask))
    expanded_polygons = align_expanded_polygon_to_original(
        expanded_polygons, tformed_polygons
    )

    return shapely.transform(
        expanded_polygons,
        skimage.transform.AffineTransform(translation=(minx, miny)),
    )


def align_expanded_polygon_to_original(polygon_expanded, polygon_original):
    nout, ninput = (
        shapely.get_num_geometries(polygon_expanded),
        shapely.get_num_geometries(polygon_original),
    )
    if nout != ninput:
        warnings.warn(
            f"Number of expanded polygons ({nout}) does not match number of input "
            f"polygons ({ninput})",
            RuntimeWarning,
        )

    if nout <= 1:
        return polygon_expanded
    # reorder multipolygon using input polygon
    tree = shapely.STRtree([pp.buffer(2) for pp in shapely.get_parts(polygon_expanded)])
    input_idx, sort_idx = tree.query(
        shapely.get_parts(polygon_original), predicate="within"
    )

    if len(input_idx) > np.unique(sort_idx).shape[0]:
        warnings.warn(
            "NN mismatch when reordering expanded polygons against input polygon",
            RuntimeWarning,
        )
    return shapely.MultiPolygon(shapely.get_parts(polygon_expanded)[sort_idx].tolist())


def multi_buffer_polygon(polygon, buffers):
    import itertools

    sortings = np.argsort(buffers)
    buffers = sorted(buffers)
    out = []
    for ii, (before, after) in enumerate(
        tqdm.tqdm(
            itertools.pairwise(buffers),
            desc="buffering polygon",
            total=len(buffers) - 1,
        )
    ):
        if ii == 0:
            out.append(polygon.buffer(before))
        out.append(out[-1].buffer(after - before))
    return np.asarray(out)[sortings]
