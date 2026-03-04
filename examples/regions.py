"""
Demonstrates the different layout options for positioning a view on a canvas.

Each layout is described by x_start, x_end, y_start, y_end Dim values using
fr() for fractional and px() for pixel units. Click anywhere on the view to
cycle through the examples. The active layout is printed to the terminal.
Resize the window to see how each configuration responds.
"""

import numpy as np

import scenex as snx
import scenex.app.events as events
from scenex.model._layout import AnyDim, fr, px
from scenex.utils.projections import zoom_to_fit

try:
    from imageio.v3 import imread

    data = np.asarray(imread("imageio:astronaut.png")).astype(np.uint8)[::-1, :, :]
except Exception:
    data = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        data[i, :, 0] = i
        data[i, :, 2] = 255 - i
    for j in range(256):
        data[:, j, 1] = j


REGIONS: list[tuple[str, AnyDim, AnyDim, AnyDim, AnyDim]] = [
    # --- Fractional: both axes (proportional placement) ---
    (
        "Fractional full canvas:",
        fr(0),
        fr(1),
        fr(0),
        fr(1),
    ),
    (
        "Fractional top-left quarter:",
        fr(0),
        fr(0.5),
        fr(0),
        fr(0.5),
    ),
    (
        "Fractional middle ninth:",
        fr(1 / 3),
        fr(2 / 3),
        fr(1 / 3),
        fr(2 / 3),
    ),
    (
        "Fractional left half, full height:",
        fr(0),
        fr(0.5),
        fr(0),
        fr(1),
    ),
    (
        "Fractional right half, full height:",
        fr(0.5),
        fr(1),
        fr(0),
        fr(1),
    ),
    # --- Pixel: absolute pixel placement ---
    (
        "Pixel fixed, 400x400 at (50, 50):",
        px(50),
        px(450),
        px(50),
        px(450),
    ),
    (
        "Pixel pin to right edge, 400px wide:",
        px(-400),
        fr(1),
        px(50),
        px(450),
    ),
    (
        "Pixel pin to bottom, 400px tall:",
        px(100),
        px(500),
        px(-400),
        fr(1),
    ),
    # --- Inset from canvas edges ---
    (
        "50px inset all sides:",
        px(50),
        px(-50),
        px(50),
        px(-50),
    ),
    (
        "Stretch from (50, 50) to canvas edge:",
        px(50),
        fr(1),
        px(50),
        fr(1),
    ),
    (
        "150px left/right inset, full height:",
        px(150),
        px(-150),
        fr(0),
        fr(1),
    ),
    # --- Mixed: fractional on one axis, pixel on the other ---
    (
        "Mixed: x=left half, y=100px from top 300px tall:",
        fr(0),
        fr(0.5),
        px(100),
        px(400),
    ),
    (
        "Mixed: x=100px each side, y=middle third:",
        px(100),
        px(-100),
        fr(1 / 3),
        fr(2 / 3),
    ),
    (
        "Mixed: x=pin right 200px wide, y=top third:",
        px(-200),
        fr(1),
        fr(0),
        fr(1 / 3),
    ),
]


img = snx.Image(data=data, clims=(0, 255))
view = snx.View(
    scene=snx.Scene(children=[img]),
    camera=snx.Camera(),
)
canvas = snx.Canvas(views=[view], width=600, height=600, visible=True)

region_idx = 0


def _on_click(event: events.Event) -> bool:
    global region_idx
    if isinstance(event, events.MousePressEvent):
        region_idx = (region_idx + 1) % len(REGIONS)
        name, x_start, x_end, y_start, y_end = REGIONS[region_idx]
        view.layout.x = x_start, x_end
        view.layout.y = y_start, y_end
        print(f"[{region_idx + 1}/{len(REGIONS)}] {name}")
    return False


view.set_event_filter(_on_click)

name, *_ = REGIONS[region_idx]
print(f"[{region_idx + 1}/{len(REGIONS)}] {name}")
print("Click the view to cycle through layout options.")

snx.show(canvas)
zoom_to_fit(view)
snx.run()
