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
  OME-TIFF
    → channel select
    → resize to target pixel size         [lazy, map_overlap, INTER_AREA]
    → [optional] rolling-ball BG subtract [lazy, rolling_ball_large.py]
    → Gaussian smooth (sigma)             [eager, small array]
    → threshold (value)                   [eager]
    → remove_small_holes (area_threshold) [eager]
    → remove_small_objects (min_size)     [eager]
    → channel mask (napari Labels layer)

Combine masks:
  [mask_A] op [mask_B] op [mask_C] ...   [left-to-right, configurable ops]
    → [optional] remove_small_holes / remove_small_objects
    → final mask (napari Labels layer)

Record parameters → YAML/JSON sidecar

Write final mask to disk
  → GeoJSON polygon  (QuPath-compatible)
  or
  → binary OME-TIFF / Zarr
```

---

## Key decisions

### 1. Do not use pre-built OME-TIFF pyramids for downsampling

Pre-built pyramid levels in OME-TIFF files are commonly generated with bilinear or
nearest-neighbor interpolation (OpenSlide, common converters), which introduces aliasing.
For quantitative fluorescence thresholding, aliasing can suppress thin bright structures
(capillaries, ducts) and create phantom signal near Nyquist.

**Decision:** always downsample from the full-resolution layer using `cv2.INTER_AREA`,
which is the correct anti-aliasing method for downscaling.

### 2. Lazy resize via `map_overlap`, not `map_blocks`

For non-integer scale factors, `cv2.INTER_AREA` has a kernel support of `ceil(1/s)` input
pixels. Naive `map_blocks` produces 1-pixel seam artifacts at chunk boundaries.

**Decision:** use `da.map_overlap` with:
- `depth = ceil(1 / scale)` input pixels on each side
- `trim=False` — shape changes so dask cannot auto-trim
- manual output trim inside the block function: `d = max(1, round(depth * scale))`

For typical mIF scales (~0.325 µm → 10 µm, scale ≈ 0.033), `depth ≈ 31` input pixels and
`d = 1` output pixel. Overhead is negligible.

For integer scale factors, `map_blocks` with chunk sizes divisible by `sf` is exact and
slightly simpler, but `map_overlap` is used uniformly for correctness.

*Practical note:* at 20–30× downscale, downstream Gaussian smoothing buries the 1-pixel
seam anyway. But implementing it properly costs little extra.

### 3. Cache the downsampled image as in-memory zarr

After the resize (and optional BG subtraction) is computed once, store it as a zarr array.
This cache is shared across:
- rolling-ball background estimation (needs random-access blocks)
- mask pipeline (smooth → threshold → cleanup)
- napari visualization

A minimal 2-level pyramid is built from the cache for napari multiscale display:
```python
napari_pyramid = [cache, cache[::2, ::2]]
viewer.add_image(napari_pyramid, multiscale=True)
```

**Options (expose in UI):**
- In-memory zarr (default): fast, gone on session exit
- Temp-file zarr: survives kernel restart, useful when BG subtraction is slow

### 4. Threshold widget

`magicgui.widgets.FloatSlider(readout=True, tracking=True)` already provides a
slider + editable spinbox combination. The readout is a bidirectional `QDoubleSpinBox`
(source: `magicgui/backends/_qtpy/widgets.py`). `tracking=True` enables live preview
as the slider is dragged.

No custom widget needed.

### 5. Multi-mask logical expression: dynamic row builder

Supports 2+ masks with a dynamic list of `(op, layer)` rows evaluated left-to-right:

```
[mask: CK         ▼]
[AND ▼] [mask: ECAD     ▼]  [−]
[OR  ▼] [mask: CD31     ▼]  [−]
[+ add layer]
```

Available ops: `AND`, `OR`, `AND NOT`, `OR NOT`, `XOR`

Serialization:
```json
[
  {"layer": "CK"},
  {"op": "AND", "layer": "ECAD"},
  {"op": "OR",  "layer": "CD31"}
]
```

Parenthesized grouping (`(A | B) & ~C`) deferred until actually needed. If required later,
a text expression input can be added alongside the builder as an override.

### 6. No napari plugin framework needed

Existing napari plugin ecosystems (napari-assistant, napari-workflows) don't cover the full
pipeline (BG subtraction + multi-channel logical ops + GeoJSON export) and have unclear dask
support. Custom `magicgui` dock widgets attached from a launcher script are sufficient and
allow direct reuse of existing code (`rolling_ball_large.py`, cleanup functions).

### 7. napari-mcp

`napari-mcp` (early-stage, napari team) exposes a running napari viewer over the Model
Context Protocol — useful for agent-driven interactive exploration (add/remove layers, run
scripts, read layer state). Not a pipeline builder. Worth installing for iterating on
thresholds via conversation rather than sliders.

---

## Widget layout (3 dock panels)

### Panel 1 — Channel mask builder
| Control | Widget |
|---|---|
| Image layer | `ComboBox` (napari image layers) |
| Channel index | `SpinBox` |
| Target pixel size (µm) | `FloatSpinBox` |
| BG subtract | `CheckBox` |
| Rolling-ball radius (px) | `FloatSpinBox` (enabled when BG subtract is on) |
| Gaussian sigma | `FloatSlider(readout=True)` |
| Threshold | `FloatSlider(readout=True, tracking=True)` |
| Remove holes (area px) | `SpinBox` |
| Remove objects (area px) | `SpinBox` |
| Output layer name | `LineEdit` |
| [Build mask] | `PushButton` |
| [Clear cache] | `PushButton` |

### Panel 2 — Combine masks
| Control | Widget |
|---|---|
| Dynamic row list | `(ComboBox op, ComboBox layer, PushButton −)` × N |
| [+ Add layer] | `PushButton` |
| Remove holes | `SpinBox` |
| Remove objects | `SpinBox` |
| Output layer name | `LineEdit` |
| [Compute] | `PushButton` |

### Panel 3 — Export
| Control | Widget |
|---|---|
| Mask layer | `ComboBox` |
| Format | `RadioButtons`: GeoJSON / Zarr / OME-TIFF |
| Output path | `FileEdit` |
| [Save + log params] | `PushButton` |

---

## Parameter log schema (JSON)

```json
{
  "file": "...",
  "timestamp": "2026-05-14T...",
  "channel_masks": [
    {
      "layer_name": "CK",
      "channel_idx": 18,
      "target_px_size": 10.0,
      "bg_subtract": true,
      "rolling_ball_radius": 50,
      "gaussian_sigma": 1.0,
      "threshold": 400,
      "remove_holes_area": 10,
      "remove_objects_area": 10
    }
  ],
  "combine": [
    {"layer": "CK"},
    {"op": "AND", "layer": "ECAD"}
  ],
  "combine_cleanup": {
    "remove_holes_area": 10,
    "remove_objects_area": 10
  },
  "output_format": "geojson"
}
```

---

## Existing code to reuse

| File | Reused as |
|---|---|
| `rolling_ball_large.py` | BG subtraction backend (already lazy) |
| `01-tumor_mask_gen.py` | `remove_small_objects`, `remove_small_holes`, `mask_to_polygon` |
| `01-tumor_mask_gen.py` | GeoJSON export logic |

---

## Open questions / deferred

- Whether to support OME-Zarr inputs in addition to OME-TIFF (straightforward with
  `da.from_zarr`, but needs testing)
- Exact output chunk spec for `map_overlap` when shape changes — needs careful dask
  `chunks=` declaration
- napari-mcp installation and wiring
