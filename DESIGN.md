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
Input loading:
  CLI            : launch.py path.ome.tif [--channels …] [--channel-names …]
  Drag-and-drop  : drop a palom-readable file → confirmation dialog (type /
                   pixel size / channels) → layers (see "Drag-and-drop loading")

For each channel mask:
  OME-TIFF (palom OmePyramidReader)
    → [optional] channel subset (--channels / --channel-names CLI args)
    → [optional] rolling-ball BG subtract
        Preview : lazy dask array (on-demand, no disk write)
        Cache   : computed in a background thread, written as a 3-level
                  multiscale zarr pyramid (see "Multiscale pyramid caching")
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
  Export runs in a background thread; a QMessageBox reports success/failure.
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

Block dimensions that aren't a multiple of the shrink factor are edge-padded
before min-pooling (and cropped after upsampling). Otherwise `_min_pool_2d`
drops the bottom/right `< sf` remainder and the bilinear upsample stretches the
background across it, under-subtracting and leaving a brighter strip on the
image's bottom/right edge. (A residual ~sf/2-px flattening from `cv2.resize`
edge replication remains and is accepted.) When cached, the subtracted result is
written as a multiscale pyramid (see "Multiscale pyramid caching").

### 7. Unique label values for distinct colors

napari colors Labels layers by label value: value 1 always maps to the same color.
`_next_label(viewer)` returns the smallest positive integer not already used as the
max value in any existing Labels layer (taking the full-res level for multiscale
layers), so each finalized mask gets a distinct color without manual bookkeeping.

### 8. Combine resolution safety

The seed mask (Mask 1) defines the coordinate frame. Non-seed dropdowns are filtered
by full-res shape (level 0 for multiscale layers) to match the seed. Changing Mask 1
resets all subsequent rows to prevent stale cross-resolution pairings.

### 9. Export pixel size handling

Pixel size is pulled from `layer.scale[-1]` when the mask layer is selected:
- **GeoJSON**: pixel_size is a load-bearing coordinate transform
  (`shapely.affinity.scale`); user may adjust before saving.
- **Zarr / OME-TIFF**: pixel_size is metadata only (zarr attrs / OME-XML); locked
  to the layer value.

### 10. Multiscale pyramid caching (shared builder)

Cached arrays — the rolling-ball result, and dropped non-pyramidal masks/images —
are written as a 3-level multiscale zarr group (levels `"0"/"1"/"2"`, each a 4×
downsample) via `mask_tool/pyramid.write_pyramid_group`. napari then renders
zoomed-out views from the small coarse levels instead of pulling full-res chunks
of a single-scale array.

Levels are written one at a time, each read back from disk before producing the
next, so an expensive level-0 graph (e.g. the rolling ball) is computed once and
peak memory stays bounded. Each level is downsampled per dask block; rechunking
the input to `factor × output_chunk` keeps chunk edges on multiples of `factor`,
so a per-block resize is bit-identical to a whole-image resize (no seams).
Interpolation is `INTER_AREA` for intensity and nearest for label masks (averaging
label IDs is meaningless). The nearest path uses **strided slicing** (`block[::f,
::f]`) rather than `cv2.resize`, which is dtype-agnostic (cv2 rejects uint32/uint64)
and exactly nearest with matching chunk sizes; masks use a zstd compressor.

Widgets read level 0 of a multiscale layer through a `_level0()` helper
(`_get_layer_data_2d`, `_compute_mask_transform`, `_pick_preview_level`, Combine,
Export, `_next_label`), so dropped multiscale masks work everywhere.

### 11. Drag-and-drop loading

Dropping a file onto napari's default readers bypasses palom: eager full-res load
and wrong pixel size (scale 1.0), which corrupts every physical-units result.
`mask_tool/dnd.py` installs a Qt event filter on the QtViewer that intercepts
drops of palom-readable files (`.ome.tif/.tif/.qptiff/.svs/.vsi`) and routes them
through a confirmation dialog; other extensions fall through to napari.

The flow is deferred: a palom reader is built for metadata only, the type is
auto-detected, and the dialog lets the user confirm/override type, pixel size, and
channels (with a name filter, row-click toggle, select/deselect-all; channels
default unchecked). Only on **Add** are layers created (**Cancel** discards the
reader). The reader is stashed on `layer.metadata` so lazy reads keep working.

Type auto-detection (overridable):
- 3-channel uint8 → **RGB**, a single `(H,W,C)` image layer (channel axis moved
  last, no channel split, no channel selection).
- 1-channel → **mask vs intensity image**, decided by foreground equality (see 12).
- otherwise → **multi-channel image** (`channel_axis` split, reversed order/names
  like the CLI loader).

Masks are added as Labels and **always** cached as a nearest-downsampled pyramid
(strided slicing, not `cv2.resize`, which rejects the uint32/uint64 dtypes common
for instance-label masks). Reusing palom's levels is *not* safe: palom synthesises
coarse pyramid levels by coarsening level 0 (chunk sizes halve per level), so
reading any coarse level pulls the **entire full-res level 0** — a multi-GB RAM/IO
spike (e.g. a 13.5 GB uint32 level 0 read just for the thumbnail; napari `Labels`
creation measured at ~3.9 GB → 63 MB after caching). Caching reads level 0 once in
a background thread (~1.1 GB peak at 4 workers on that mask) and builds real cheap
coarse levels. Non-pyramidal single-channel images larger than 4096 px are cached
with INTER_AREA; multi-channel / RGB images use palom's levels directly (caching
deferred), so they share palom's coarse-read cost — as does the CLI-loaded image.
The dialog's pixel size sets layer `scale`/`translate`, so all widgets (which key
off `layer.scale`) stay consistent — the partial `--px-size` / `_px_size_override`
mechanism is left unchanged and is not extended.

### 12. Mask-vs-intensity detection by foreground equality

A label mask is piecewise-constant (every pixel inside an object equals its
neighbours; only boundaries differ); an intensity image has per-pixel noise. The
discriminator is the fraction of neighbouring **foreground** pixels (both nonzero)
that are identical — ~0.9+ for masks regardless of label count or object size,
~0.01 for intensity. Excluding background is essential: a cropped FOV with large
zero regions would otherwise false-positive. The metric is computed on full-res
sample tiles located via the coarsest pyramid level (densest tissue first), so
detection ignores the glass/exterior that dominates a WSI's top-left. Threshold
0.5; falls back to the integer-dtype rule when content is too sparse to judge.
(Compression ratio was considered but rejected — it conflates piecewise-constancy
with storage bit-depth.)

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
  pyramid.py       — write_pyramid_group (shared multiscale zarr pyramid builder)
  rolling_ball.py  — subtract_background, subtract_background_lazy, rolling_ball_background
  dnd.py           — drag-and-drop: reader routing, type detection, Add dialog, install_drop_handler
  export.py        — export_geojson, export_zarr, export_tiff, save_params, mask_to_polygon
  widgets.py       — RollingBallWidget, ThresholdWidget, CombineWidget, ExportWidget, MaskInfoWidget
launch.py          — CLI entry point (--channels, --channel-names, --px-size); installs drop handler
```

---

## Resolved / closed questions

- **OME-Zarr input**: not implemented; palom handles OME-TIFF well and that covers all
  current samples.
- **map_overlap chunk shape**: resolved by switching to aligned map_blocks instead.
- **napari-mcp**: not yet installed; deferred.
- **Threshold widget**: contrast_limits approach is more idiomatic in napari than a
  separate slider and requires no custom widget.
- **Mask-vs-intensity detection metric**: foreground adjacent-equality chosen over
  compression ratio (which conflates piecewise-constancy with storage bit-depth).

## Open / deferred

- **High RAM during/after rolling-ball cache**: not a leak — the spike is the
  parallel compute working set (`num_workers × per-block float32`, all cores for
  radius > 10), retained by the allocator and plateauing across runs. Fix later by
  capping `num_workers` or running the compute in a subprocess (the on-disk zarr is
  already the hand-off).
- **palom synthesises pyramids**: `OmePyramidReader` derives coarse levels by
  coarsening level 0, so reading any coarse level pulls the full-res level 0. Masks
  work around this by always caching (decision 11), but **dropped RGB / multi-channel
  images and the CLI-loaded image still use palom's levels directly** and therefore
  re-read level 0 for coarse views. Caching them is deferred (the shared builder is
  2-D only; would need to handle the channel axis). A cleaner fix would read the
  TIFF's *stored* pyramid IFDs (e.g. via tifffile) instead of palom's synthesised
  ones.
- **Dropped masks are view-only**: dask/zarr-backed Labels aren't paintable; fine
  for combine/export, which is the intent.
- **Drop handler hook point**: filter is on `viewer.window._qt_viewer`; may need to
  move to the vispy canvas child depending on napari version (verify in the app).
