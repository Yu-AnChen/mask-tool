# ---------------------------------------------------------------------------- #
#                                 polygon_util                                 #
# ---------------------------------------------------------------------------- #
import warnings

import cv2
import numpy as np
import shapely
import skimage.segmentation
import skimage.transform
import tqdm
import zarr


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


# ---------------------------------------------------------------------------- #
#                              End of polygon_util                             #
# ---------------------------------------------------------------------------- #


import functools
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import shapely
import shapely.affinity
import tqdm
from rc_pop_density.density import parse_roi_geometry
from rc_pop_density.populations import parse_csv
from shapely.geometry import shape
from shapely.ops import unary_union


def get_exclude_geom(populations: dict):
    """
    Parse all Exclude_(...) shape entries in the defs CSV and return their union.
    Relies on the Exclude Area definition to identify which shapes are excludes,
    but falls back to collecting any key starting with 'Exclude_('.
    """
    exclude_names = [name for name in populations if name.startswith("Exclude_(")]
    geoms = []
    for name in exclude_names:
        definition = populations[name]
        if not definition:
            print(f"  [warn] {name} has no coordinate definition", file=sys.stderr)
            continue
        geom = parse_roi_geometry(definition)
        if geom is None:
            print(f"  [warn] could not parse geometry for {name}", file=sys.stderr)
            continue
        print(f"  {name}: {geom.geom_type}, area={geom.area:.0f} px²")
        geoms.append(geom)

    if not geoms:
        return None
    return unary_union(geoms)


def load_roi_geom(geojson_path: str):
    """Load the first (or union of all) features from a QuPath geojson export."""
    with open(geojson_path) as f:
        data = json.load(f)
    features = data["features"] if "features" in data else [data]
    geoms = [shape(feat["geometry"]) for feat in features]
    roi = unary_union(geoms) if len(geoms) > 1 else geoms[0]
    print(f"  ROI: {roi.geom_type}, area={roi.area:.0f} px²")
    return roi


def to_multipolygon(geom):
    return shapely.MultiPolygon(
        [gg for gg in geom.geoms if isinstance(gg, shapely.Polygon)]
    )


PX_SIZE = 0.325
to_um = functools.partial(
    shapely.affinity.scale, xfact=PX_SIZE, yfact=PX_SIZE, origin=(0, 0)
)


files = r"""slide_id,dir,sc,defs,roi_tissue,roi_tumor,img
Q127_S307,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S307_or19_A107_full_001\Q127_S307_012650,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S307_or19_A107_full_001\Q127_S307_012650\Population_Clean Area_2605112205.csv,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S307_or19_A107_full_001\Q127_S307_012650\Population_Clean Area_defs_2605112205.csv,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_NucleAI_samples\QuPath-ROI\ROI-tissue\Q127_S307_or19_A107_full_001_Q127_S307_012650.geojson,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_NucleAI_samples\QuPath-ROI\ROI-tumor-v3\Q127_S307_or19_A107_full_001_Q127_S307_012650.geojson,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S307_or19_A107_full_001\Q127_S307_012650\Q127_S307_or19_A107_full_001_Q127_S307_012650.ome.tiff
Q127_S314,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S314_or19_A107_full_001\Q127_S314_012651,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S314_or19_A107_full_001\Q127_S314_012651\Population_Clean Area_2605112220.csv,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S314_or19_A107_full_001\Q127_S314_012651\Population_Clean Area_defs_2605112220.csv,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_NucleAI_samples\QuPath-ROI\ROI-tissue\Q127_S314_or19_A107_full_001_Q127_S314_012651.geojson,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_NucleAI_samples\QuPath-ROI\ROI-tumor-v3\Q127_S314_or19_A107_full_001_Q127_S314_012651.geojson,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S314_or19_A107_full_001\Q127_S314_012651\Q127_S314_or19_A107_full_001_Q127_S314_012651.ome.tiff
Q127_S319,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S319_or19_A107_full_001\Q127_S319_012652,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S319_or19_A107_full_001\Q127_S319_012652\Population_Clean Area_2605112241.csv,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S319_or19_A107_full_001\Q127_S319_012652\Population_Clean Area_defs_2605112241.csv,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_NucleAI_samples\QuPath-ROI\ROI-tissue\Q127_S319_or19_A107_full_001_Q127_S319_012652.geojson,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_NucleAI_samples\QuPath-ROI\ROI-tumor-v3\Q127_S319_or19_A107_full_001_Q127_S319_012652.geojson,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S319_or19_A107_full_001\Q127_S319_012652\Q127_S319_or19_A107_full_001_Q127_S319_012652.ome.tiff
Q127_S324,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S324_or19_A107_full_001\Q127_S324_012653,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S324_or19_A107_full_001\Q127_S324_012653\Population_Clean Area_2605112251.csv,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S324_or19_A107_full_001\Q127_S324_012653\Population_Clean Area_defs_2605112251.csv,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_NucleAI_samples\QuPath-ROI\ROI-tissue\Q127_S324_or19_A107_full_001_Q127_S324_012653.geojson,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_NucleAI_samples\QuPath-ROI\ROI-tumor-v3\Q127_S324_or19_A107_full_001_Q127_S324_012653.geojson,\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_S324_or19_A107_full_001\Q127_S324_012653\Q127_S324_or19_A107_full_001_Q127_S324_012653.ome.tiff
"""


import io
import csv

tasks = list(csv.DictReader(io.StringIO(files)))

for tt in tasks[:]:
    SLIDE_ID = tt["slide_id"]
    path_defs = tt["defs"]
    path_tissue = tt["roi_tissue"]
    path_tumor = tt["roi_tumor"]
    path_sc_table = tt["sc"]

    # load exclusion ROI
    populations = parse_csv(path_defs)
    geom_exclude = get_exclude_geom(populations)

    # load and finalize tissue ROI
    _geom_tissue = load_roi_geom(path_tissue)
    if geom_exclude is not None:
        _geom_tissue -= geom_exclude
    geom_tissue = to_multipolygon(_geom_tissue)

    # load and finalize tumor ROI
    _geom_tumor = load_roi_geom(path_tumor)
    _geom_tumor &= geom_tissue
    geom_tumor = to_multipolygon(_geom_tumor)

    # convert to µm
    geom_tumor = to_um(geom_tumor)
    geom_tissue = to_um(geom_tissue)

    df = pd.read_csv(path_sc_table, engine="pyarrow")
    coords = (
        np.array(
            shapely.get_coordinates(df["centroid"].map(lambda x: shapely.from_wkt(x)))
        )
        * PX_SIZE
    )

    # W, H = np.ceil(coords.max(axis=1)).astype("int")
    raster_tumor, (row_st, col_st) = polygons_to_mask(geom_tumor, binarize=True)

    import scipy.ndimage as ndi

    expansions = np.arange(0, 101, 10)
    pad = max(expansions) + 10
    _raster_tumor = np.pad(raster_tumor, pad, mode="constant", constant_values=0)
    raster_dist = np.empty(_raster_tumor.shape, dtype="float32")
    cv2.distanceTransform(
        _raster_tumor - 1,
        distanceType=cv2.DIST_L2,
        maskSize=cv2.DIST_MASK_PRECISE,
        dst=raster_dist,
    )
    _coords = coords - [col_st, row_st] + pad
    # order=0 (nearest) seems to be more "precise"
    dist = ndi.map_coordinates(
        raster_dist, np.fliplr(_coords).T, mode="nearest", order=0
    )

    areas = []
    for ee in tqdm.tqdm(expansions):
        raster_expanded = (raster_dist[pad:, pad:] <= ee).astype("uint8")
        geom_expanded = shapely.affinity.translate(
            mask_to_polygon(raster_expanded), xoff=col_st, yoff=row_st
        )
        areas.append(geom_expanded.intersection(geom_tissue).area)

    # FIXME should parse from defs CSV
    celltypes_to = "CD163,CD20,CD31,CD3e,CD4,CD45,CD45RO,CD68,CD8a,E-cad,FOXP3,Ki-67,Pan-CK,PD-1,PD-L1,SMA,Vimentin,b_cell,cd4_t_cell,cd4_treg_cell,cd8_t_cell,CTL,Macrophage,macrophage_cell,Thelper,Treg".split(
        ","
    )

    celltypes_to = [f"Population_{tt}" for tt in celltypes_to]
    df_count = df[celltypes_to].copy()
    df_count["All cells"] = True

    df_out = pd.DataFrame(index=df_count.columns, columns=expansions)
    for ee in tqdm.tqdm(expansions):
        gb = df_count.groupby((dist <= ee) & df["Population_Clean Area"].values)
        df_out[ee] = gb.sum().loc[True]
    df_out["inf"] = df_count.groupby(df["Population_Clean Area"]).sum().loc[True]
    df_out["Clean Area"] = df_out["inf"]

    df_out.iloc[:, 1:-1] -= df_out.iloc[:, 0].values.reshape(-1, 1)
    col_names = {
        nn: "Within tumor" if nn == 0 else f"Peritumoral: {nn} um"
        for nn in df_out.columns[:-1]
    }
    df_out.rename(columns=col_names, inplace=True)

    # to mm^2
    aareas = np.array(areas + [geom_tissue.area] * 2) / 1_000_000
    aareas[1:-1] -= aareas[0]
    df_out_density = df_out / aareas
    df_out_density = df_out_density.round(2)

    df_out_area = pd.DataFrame(columns=df_out.columns)
    df_out_area.loc["Area (um2)"] = aareas

    out_dir = r"\\rc-lab-store-4\RC-LAB-STORE-5\orion\rcpnl\orion19\scans\Q127_NucleAI_samples\Peritumoral-density"
    out_dir_csv = pathlib.Path(out_dir) / "csv"
    out_dir_csv.mkdir(parents=True, exist_ok=True)
    out_dir_excel = pathlib.Path(out_dir) / "excel"
    out_dir_excel.mkdir(parents=True, exist_ok=True)

    df_out.to_excel(out_dir_excel / f"{SLIDE_ID}-peritumoral-count.xlsx")
    df_out_density.to_excel(out_dir_excel / f"{SLIDE_ID}-peritumoral-density.xlsx")
    df_out_area.to_excel(out_dir_excel / f"{SLIDE_ID}-peritumoral-area.xlsx")

    df_out.to_csv(out_dir_csv / f"{SLIDE_ID}-peritumoral-count.csv")
    df_out_density.to_csv(out_dir_csv / f"{SLIDE_ID}-peritumoral-density.csv")
    df_out_area.to_csv(out_dir_csv / f"{SLIDE_ID}-peritumoral-area.csv")
