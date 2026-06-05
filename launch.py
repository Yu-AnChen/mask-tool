"""
Launch the napari mask-building tool.

Usage
-----
  pixi run python launch.py
  pixi run python launch.py path/to/image.ome.tiff
  pixi run python launch.py path/to/image.ome.tiff --channels 0 2 5
  pixi run python launch.py path/to/image.ome.tiff --channel-names DAPI CD31 aSMA
  pixi run python launch.py path/to/image.ome.tiff --channels 0 2 5 --channel-names DAPI CD31 aSMA
  pixi run python launch.py path/to/image.ome.tiff --out-dir /path/to/output --id sample01

The image is loaded as a multiscale dask array via palom so no data is read
into RAM until a mask is built.
"""

from __future__ import annotations

import pathlib
import signal
import sys
import argparse

import napari


def _load_image(
    path: str,
    viewer: napari.Viewer,
    channels: list[int] | None = None,
    channel_names_override: list[str] | None = None,
) -> None:
    import palom.reader
    import ome_types

    reader = palom.reader.OmePyramidReader(path)
    n_ch = reader.pyramid[0].shape[0]

    try:
        channel_names = [
            cc.name
            for cc in ome_types.from_tiff(path).images[0].pixels.channels
        ]
    except Exception:
        channel_names = [f"ch{i}" for i in range(n_ch)]

    # validate + apply --channels
    if channels is not None:
        out_of_range = [i for i in channels if i < 0 or i >= n_ch]
        if out_of_range:
            sys.exit(
                f"Error: --channels indices {out_of_range} are out of range "
                f"(image has {n_ch} channels, valid indices: 0–{n_ch - 1})"
            )
        pyramid = [level[channels] for level in reader.pyramid]
        channel_names = [channel_names[i] for i in channels]
    else:
        pyramid = reader.pyramid

    # validate + apply --channel-names
    if channel_names_override is not None:
        # when --channels is absent the override must cover all channels
        expected = len(channels) if channels is not None else n_ch
        if len(channel_names_override) != expected:
            sys.exit(
                f"Error: --channel-names has {len(channel_names_override)} "
                f"name(s) but {'selected' if channels is not None else 'image'} "
                f"has {expected} channel(s)"
            )
        channel_names = channel_names_override

    px = reader.pixel_size
    # Reverse channel axis so ch 0 lands on top of the layer list
    # (napari adds layers sequentially; the last-added ends up on top)
    viewer.add_image(
        [level[::-1] for level in pyramid],
        channel_axis=0,
        name=channel_names[::-1],
        multiscale=True,
        visible=False,
        contrast_limits=(0, 5000),
        scale=(px, px),
        translate=(px * 0.5, px * 0.5),
    )
    viewer.title = pathlib.Path(path).stem
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    print(f"Loaded {path}")
    print(f"  pixel size : {px} µm")
    print(f"  channels   : {channel_names}")
    print(f"  shape (L0) : {reader.pyramid[0].shape}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="napari mask builder")
    parser.add_argument("image", nargs="?", help="Path to OME-TIFF")
    parser.add_argument(
        "--channels", nargs="+", type=int, metavar="IDX",
        help="Zero-based channel indices to load (default: all)",
    )
    parser.add_argument(
        "--channel-names", nargs="+", metavar="NAME",
        help="Channel names; must match --channels count (or total channels if --channels omitted)",
    )
    parser.add_argument(
        "--out-dir", metavar="DIR",
        help="Default output directory for exported masks",
    )
    parser.add_argument(
        "--id", metavar="ID",
        help="Sample identifier; exported files default to {id}-mask.geojson / {id}-mask.ome.tif",
    )
    parser.add_argument(
        "--params", metavar="FILE",
        help="Path to a .params.json from a previous run; pre-fills widget controls for a new image",
    )
    parser.add_argument(
        "--px-size", type=float, metavar="UM",
        help="Override source pixel size (µm/px); use when OME metadata contains wrong pixel size",
    )
    args = parser.parse_args(argv)

    if args.channels is not None and args.channel_names is not None:
        if len(args.channels) != len(args.channel_names):
            parser.error(
                f"--channels ({len(args.channels)}) and --channel-names "
                f"({len(args.channel_names)}) must have the same length"
            )

    viewer = napari.Viewer(title="Mask Builder")

    if args.image:
        _load_image(
            args.image, viewer,
            channels=args.channels,
            channel_names_override=args.channel_names,
        )

    # build and attach widgets
    from mask_tool.widgets import RollingBallWidget, ThresholdWidget, CombineWidget, ExportWidget, MaskInfoWidget

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path.home()
    stem = f"{args.id}-mask" if args.id else "mask"
    default_export_path = out_dir / f"{stem}.geojson"

    rb_widget   = RollingBallWidget(viewer)
    thr_widget  = ThresholdWidget(viewer)
    comb_widget = CombineWidget(viewer)
    exp_widget  = ExportWidget(viewer, threshold_widget=thr_widget, combine_widget=comb_widget,
                               default_path=default_export_path)
    info_widget = MaskInfoWidget(viewer)

    viewer.window.add_dock_widget(rb_widget,   area="right", name="BG subtraction")
    viewer.window.add_dock_widget(thr_widget,  area="right", name="Threshold", tabify=True)
    viewer.window.add_dock_widget(comb_widget, area="right", name="Combine masks")
    viewer.window.add_dock_widget(exp_widget,  area="right", name="Export", tabify=True)
    viewer.window.add_dock_widget(info_widget, area="right", name="Mask info")

    if args.px_size:
        thr_widget.set_px_size_override(args.px_size)
        print(f"  px override: {args.px_size} µm (--px-size)")

    # Route drag-and-drop of palom-readable files through a dialog so layers get
    # the correct pixel size / type instead of napari's default eager reader.
    from mask_tool.dnd import install_drop_handler
    install_drop_handler(viewer, default_px_size=args.px_size)

    if args.params:
        import json
        params_path = pathlib.Path(args.params)
        if params_path.exists():
            with open(params_path) as f:
                params_data = json.load(f)
            rb_widget.load_session_params(params_data)
            thr_widget.load_session_params(params_data)
            comb_widget.load_session_params(params_data)
            print(f"Loaded params: {params_path}")
        else:
            print(f"Warning: --params file not found: {params_path}")

    if args.image:
        print(f"  cache dir  : {rb_widget._cache_dir}")

    # Qt's C++ event loop doesn't deliver Python signals unless the interpreter
    # gets control periodically; the no-op timer wakes it every 200 ms so
    # Ctrl+C reaches the handler and viewer.close() can clean up normally.
    from qtpy.QtCore import QTimer
    _sigint_timer = QTimer()
    _sigint_timer.start(200)
    _sigint_timer.timeout.connect(lambda: None)
    signal.signal(signal.SIGINT, lambda *_: viewer.close())

    napari.run()


if __name__ == "__main__":
    main(sys.argv[1:])
