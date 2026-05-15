"""
Launch the napari mask-building tool.

Usage
-----
  pixi run python launch.py
  pixi run python launch.py path/to/image.ome.tiff
  pixi run python launch.py path/to/image.ome.tiff --channel 10 --px-size 0.325

The image is loaded as a multiscale dask array via palom so no data is read
into RAM until a mask is built.
"""

from __future__ import annotations

import pathlib
import signal
import sys
import argparse

import napari


def _load_image(path: str, viewer: napari.Viewer) -> None:
    import palom.reader
    import ome_types

    reader = palom.reader.OmePyramidReader(path)
    try:
        channel_names = [
            cc.name
            for cc in ome_types.from_tiff(path).images[0].pixels.channels
        ]
    except Exception:
        channel_names = [f"ch{i}" for i in range(reader.pyramid[0].shape[0])]

    px = reader.pixel_size
    viewer.add_image(
        reader.pyramid,
        channel_axis=0,
        name=channel_names,
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
    args = parser.parse_args(argv)

    viewer = napari.Viewer(title="Mask Builder")

    if args.image:
        _load_image(args.image, viewer)

    # build and attach widgets
    from mask_tool.widgets import RollingBallWidget, ThresholdWidget, CombineWidget, ExportWidget, MaskInfoWidget

    rb_widget   = RollingBallWidget(viewer)
    thr_widget  = ThresholdWidget(viewer)
    comb_widget = CombineWidget(viewer)
    exp_widget  = ExportWidget(viewer, threshold_widget=thr_widget, combine_widget=comb_widget)
    info_widget = MaskInfoWidget(viewer)

    viewer.window.add_dock_widget(rb_widget,   area="right", name="BG subtraction")
    viewer.window.add_dock_widget(thr_widget,  area="right", name="Threshold")
    viewer.window.add_dock_widget(comb_widget, area="right", name="Combine masks")
    viewer.window.add_dock_widget(exp_widget,  area="right", name="Export")
    viewer.window.add_dock_widget(info_widget, area="right", name="Mask info")

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
