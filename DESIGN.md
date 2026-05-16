# napari mIF Mask Tool — Design Notes

## Context

`01-tumor_mask_gen.py` implements a threshold-based tumor mask pipeline for multiplexed
immunofluorescence (mIF) whole-slide images (OME-TIFF). It reads CK and E-cadherin channels,
downsamples to 10 µm/px, applies Gaussian smoothing + thresholding + morphological cleanup,
and exports GeoJSON polygons loadable into QuPath.

The goal of this tool is to generalize that workflow into an interactive napari-based pipeline
that is lazily evaluated (dask + zarr), parameterized, and reusable across channels and samples.

---

## Pipeline stages

```
For each channel mask:
  OME-TIFF (palom OmePyramidReader)
    → [optional] channel subset (--channels / --channel-names CLI args)
    → [optional] rolling-ball BG subtract
        Preview : lazy dask array (on-demand, no disk write)
        Cache   : computed to zarr on disk in a background thread
    → downsample to target pixel size (background thread)
        lazy_resize: map_blocks with chunks aligned to INTER_AREA boundaries
        stored as numpy array (_preview_raw) for fast re-blur
    → Gaussian smooth (sigma, cv2.GaussianBlur)
        applied to _preview_raw; debounced live update
    → threshold via napari contrast_limits lower bound
        live readout in integer or float depending on dtype
    → finalize (background thread):
        mask = preview_data > threshold
        remove_small_holes  (area ≤ N µm²)
        remove_small_objects (area < N µm²)
    → Labels layer with unique label value (distinct napari color per layer)

Combine masks:
  [mask_A] op [mask_B] op [mask_C] ...  (left-to-right, configurable ops)
    → computed in background thread
    → [optional] remove_small_holes / remove_small_objects
    → Labels layer

Record parameters → JSON sidecar
  filename mirrors mask file with dots→underscores + "-params.json"
  e.g. mask.ome.tif → mask_ome_tif-params.json
       mask.geojson → mask_geojson-params.json

Write final mask to disk:
  → GeoJSON polygon    (QuPath-compatible; coordinates scaled by pixel_size)
  → GeoJSON (zip)      (deflate-compressed .geojson.zip)
  → OME-TIFF           (palom pyramidal, is_mask=True, zlib, .ome.tif)
  → Zarr               (binary bool, pixel_size stored in attrs)

  File picker dialog triggers save immediately.
  "Save mask and params" button re-saves to the current path.
  napari show_info notification on completion.
```

---

## Dock widget layout (5 panels)

### Panel 1 — BG subtraction (RollingBallWidget)
| Control | Widget |
|---|---|
| Image layer | `ComboBox` (auto-tracks selection) |
| Ball radius | `DoubleSpinBox` (µm) |
| [Preview] | Lazy layer, instant |
| [Cache] | Background thread → zarr on disk |
| Cache dir | `LineEdit` + browse + clear |

### Panel 2 — Threshold (ThresholdWidget)
| Control | Widget |
|---|---|
| Image layer | `ComboBox` |
| Source px size | `DoubleSpinBox` (µm/px, auto-populated from layer) |
| Mask px size | `DoubleSpinBox` (µm/px) |
| Gaussian sigma | `DoubleSpinBox` (px, debounced live re-blur) |
| Invert colormap | `CheckBox` (brightfield mode) |
| [Add Preview Layer] | Downsample in background thread; adds `pre-{name}` layer |
| Threshold readout | `QLabel` — tracks contrast_limits[0] of preview layer |
| Fill holes ≤ | `SpinBox` (µm²) |
| Remove objects < | `SpinBox` (µm², default 0) |
| [Finalize Mask] | Background thread → adds `fin-{name}` Labels layer |

### Panel 3 — Combine masks (CombineWidget)
| Control | Widget |
|---|---|
| Mask 1 | `ComboBox` (all Labels layers) |
| [op ▾] Mask N | `ComboBox` op + `ComboBox` layer (filtered to seed shape) + [−] |
| [+ Add row] | Appends a row; non-seed dropdowns show only shape-matched layers |
| Fill holes ≤ | `SpinBox` (µm²) |
| Remove objects < | `SpinBox` (µm²) |
| Output name | `LineEdit` |
| [Finalize combined mask] | Background thread → Labels layer |

Seed mask change resets all subsequent rows for a clean state.

### Panel 4 — Export (ExportWidget)
| Control | Widget |
|---|---|
| Mask layer | `ComboBox` (Labels layers) |
| Pixel size | `DoubleSpinBox` — auto-populated from layer; editable only for GeoJSON |
| Format | `ComboBox`: GeoJSON / GeoJSON (zip) / Zarr / TIFF |
| Output path | `LineEdit` + [browse + save] |
| [Save mask and params] | Re-save to current path without file picker |

### Panel 5 — Mask info (MaskInfoWidget)
Displays `mask_params` metadata for the currently selected Labels layer as a
key/value grid. Updated on selection change and layer insertion.

---

## Key design decisions

### 1. Do not use pre-built OME-TIFF pyramids for downsampling

Pre-built pyramid levels are commonly generated with bilinear or nearest-neighbor
interpolation, which introduces aliasing. For quantitative fluorescence thresholding,
aliasing can suppress thin bright structures (capillaries, ducts).

**Decision:** always downsample from full resolution using `cv2.INTER_AREA`.

### 2. Lazy resize via `map_blocks` with aligned chunk sizing

For non-integer scale factors, `cv2.INTER_AREA` has fractional kernel boundaries at
chunk edges, causing seam artifacts with naive `map_blocks`.

**Solution** (in `mask_tool/resize.py`): express scale as `p/q` (lowest terms via
`Fraction.limit_denominator(1000)`). Align chunk sizes to multiples of `q`. At this
alignment, INTER_AREA kernel boundaries fall exactly on chunk edges — each chunk's
output is pixel-identical to the same region in a full-image resize. No overlap
padding is needed; no seams.

Guard: `cv2.resize` errors when output dimension rounds to 0. Remainder chunks
smaller than `ceil(0.5/scale)` are absorbed into the preceding chunk.

The original design specified `map_overlap`; the aligned `map_blocks` approach is
simpler, cheaper, and produces identical results.

### 3. Threshold via napari contrast limits

No custom slider widget. Instead:
- The preview Image layer uses a threshold colormap (black→white or inverted).
- The lower contrast limit acts as the threshold value.
- A `contrast_limits` event listener updates the threshold readout label.
- The readout is cast to `int` when the preview data has an integer dtype, keeping
  the displayed and logged values in sync.

### 4. Morphological operations — pure OpenCV

`remove_small_objects` and `remove_small_holes` in `mask_tool/pipeline.py` use
`cv2.connectedComponentsWithStats` instead of scikit-image. This eliminates the
scikit-image dependency for mask cleanup and keeps the operations consistent across
the single-channel and combine paths.

### 5. Background threading

All heavy compute runs in napari `@thread_worker` background threads:
- Rolling ball cache computation
- Image downsampling (lazy_resize → .compute())
- Mask finalization (threshold + hole fill + object removal)
- Mask combination

Buttons are disabled for the duration and re-enabled on completion or error.

### 6. Rolling-ball background subtraction

Implemented in `mask_tool/rolling_ball.py`, mirroring imagec's shrink→roll→enlarge
strategy. Min-pools by a shrink factor (sf = 1/2/4/8 depending on radius), runs
skimage rolling ball at the reduced scale, bilinear-upsamples the estimated
background, then subtracts.

Thread allocation is tuned per regime: for sf ≥ 2 the shrunk image is tiny, so
OpenMP overhead dominates — all cores go to dask parallelism and OpenMP is
disabled per block. For sf = 1 cores are split evenly between dask and OpenMP.

### 7. Unique label values for distinct colors

napari colors Labels layers by label value: value 1 always maps to the same color.
`_next_label(viewer)` returns the smallest positive integer not already used as the
max value in any existing Labels layer, so each finalized mask gets a distinct color
without manual bookkeeping.

### 8. Combine resolution safety

The seed mask (Mask 1) defines the coordinate frame. Non-seed dropdowns are filtered
by `data.shape` to match the seed. Changing Mask 1 resets all subsequent rows to
prevent stale cross-resolution pairings.

### 9. Export pixel size handling

Pixel size is pulled from `layer.scale[-1]` when the mask layer is selected:
- **GeoJSON**: pixel_size is a load-bearing coordinate transform
  (`shapely.affinity.scale`); user may adjust before saving.
- **Zarr / OME-TIFF**: pixel_size is metadata only (zarr attrs / OME-XML); locked
  to the layer value.

---

## Parameter log schema (JSON)

```json
{
  "timestamp": "2026-05-15T...",
  "output_layer": "fin-CD31",
  "output_file": "/path/to/mask.ome.tif",
  "threshold_masks": {
    "fin-CD31": {
      "type": "threshold",
      "source_layer": "CD31",
      "preview_layer": "pre-CD31",
      "px_size_src_um": 0.65,
      "target_px_size_um": 4.0,
      "gaussian_sigma_px": 1.0,
      "threshold": 272,
      "hole_threshold_um2": 1000,
      "obj_threshold_um2": 300,
      "invert": false
    }
  },
  "combine": {
    "steps": [
      {"layer": "fin-CD31"},
      {"op": "AND", "layer": "fin-SMA"}
    ],
    "hole_threshold": 1000,
    "obj_threshold": 0
  }
}
```

---

## File layout

```
mask_tool/
  __init__.py
  pipeline.py      — remove_small_objects, remove_small_holes, combine_masks, CombineParams
  resize.py        — lazy_resize (map_blocks, aligned chunk sizing)
  rolling_ball.py  — subtract_background, subtract_background_lazy, rolling_ball_background
  export.py        — export_geojson, export_zarr, export_tiff, save_params, mask_to_polygon
  widgets.py       — RollingBallWidget, ThresholdWidget, CombineWidget, ExportWidget, MaskInfoWidget
launch.py          — CLI entry point (--channels, --channel-names)
```

---

## Resolved / closed questions

- **OME-Zarr input**: not implemented; palom handles OME-TIFF well and that covers all
  current samples.
- **map_overlap chunk shape**: resolved by switching to aligned map_blocks instead.
- **napari-mcp**: not yet installed; deferred.
- **Threshold widget**: contrast_limits approach is more idiomatic in napari than a
  separate slider and requires no custom widget.
