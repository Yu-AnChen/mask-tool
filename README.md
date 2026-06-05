# mask-tool

An interactive [napari](https://napari.org) tool for building binary/label masks
from whole-slide images (OME-TIFF): rolling-ball background subtraction,
threshold, combine, and export to GeoJSON / OME-TIFF / Zarr.

## Install & run

The project uses [pixi](https://pixi.sh) to manage the environment — you don't
need a pre-existing Python or conda install.

### 1. Install pixi

**macOS / Linux:**

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

Then **restart your terminal** so `pixi` is on the `PATH`. (Optional: keep pixi
up to date with `pixi self-update`.)

### 2. Install git (if you don't have it)

If `git --version` fails — common on Windows — install it with pixi:

```sh
pixi global install git
```

### 3. Clone the repository

```sh
git clone https://github.com/Yu-AnChen/mask-tool.git
cd mask-tool
```

### 4. Install dependencies

```sh
pixi install --locked
```

This creates the environment exactly from `pixi.lock` (first run downloads
everything; later runs are instant).

### 5. Launch the viewer

```sh
pixi run python launch.py
```

Or open an image directly:

```sh
pixi run python launch.py path/to/image.ome.tif
```

You can also start with `pixi run python launch.py` and **drag a file**
(OME-TIFF / TIFF / SVS / VSI) onto the viewer to load it. See
`pixi run python launch.py --help` for options (`--channels`, `--channel-names`,
`--px-size`, `--out-dir`, `--id`, `--params`).
