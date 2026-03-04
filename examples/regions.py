"""
Demonstrates the different span types for positioning a view on a canvas.

Each layout is described by an independent x_span and y_span. Click anywhere
on the view to cycle through the examples. The active spans are printed to the
terminal. Resize the window to see how each configuration responds.
"""

import numpy as np

import scenex as snx
import scenex.app.events as events
from scenex import Fractional, OffsetPlusSize, PixelGaps
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


REGIONS: list[tuple[str, snx.Span, snx.Span]] = [
    # --- Fractional: both axes (proportional placement) ---
    (
        "Fractional full canvas:",
        Fractional(start=0, end=1, total=1),
        Fractional(start=0, end=1, total=1),
    ),
    (
        "Fractional top-left quarter:",
        Fractional(start=0, end=1, total=2),
        Fractional(start=0, end=1, total=2),
    ),
    (
        "Fractional middle ninth:",
        Fractional(start=1, end=2, total=3),
        Fractional(start=1, end=2, total=3),
    ),
    (
        "Fractional left half, full height:",
        Fractional(start=0, end=1, total=2),
        Fractional(start=0, end=1, total=1),
    ),
    (
        "Fractional right half, full height:",
        Fractional(start=1, end=2, total=2),
        Fractional(start=0, end=1, total=1),
    ),
    # --- OffsetPlusSize: both axes (absolute pixel placement) ---
    (
        "OffsetPlusSize fixed 400x400 at (50, 50):",
        OffsetPlusSize(offset=50, size=400),
        OffsetPlusSize(offset=50, size=400),
    ),
    (
        "OffsetPlusSize pin to right edge:",
        OffsetPlusSize(offset=-400, size=400),
        OffsetPlusSize(offset=50, size=400),
    ),
    (
        "OffsetPlusSize pin to bottom:",
        OffsetPlusSize(offset=100, size=400),
        OffsetPlusSize(offset=-400, size=400),
    ),
    # --- PixelGaps: both axes (gaps on each side) ---
    (
        "PixelGaps 50px margin all sides:",
        PixelGaps(left=50, right=50),
        PixelGaps(left=50, right=50),
    ),
    (
        "PixelGaps stretch from (50, 50) to canvas edge:",
        PixelGaps(left=50, right=0),
        PixelGaps(left=50, right=0),
    ),
    (
        "PixelGaps 150px left/right, 0px top/bottom:",
        PixelGaps(left=150, right=150),
        PixelGaps(left=0, right=0),
    ),
    # --- Mixed span types across axes ---
    (
        "Mixed: x=Fractional left half, y=OffsetPlusSize 100px from top size 300",
        Fractional(start=0, end=1, total=2),
        OffsetPlusSize(offset=100, size=300),
    ),
    (
        "Mixed: x=PixelGaps 100px each side, y=Fractional middle third",
        PixelGaps(left=100, right=100),
        Fractional(start=1, end=2, total=3),
    ),
    (
        "Mixed: x=OffsetPlusSize pin right 200px wide, y=Fractional top third",
        OffsetPlusSize(offset=-200, size=200),
        Fractional(start=0, end=1, total=3),
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
        name, x_span, y_span = REGIONS[region_idx]
        view.layout.x_span = x_span  # type: ignore
        view.layout.y_span = y_span  # type: ignore
        print(f"[{region_idx + 1}/{len(REGIONS)}] {name}")
    return False


view.set_event_filter(_on_click)

name, _, _ = REGIONS[region_idx]
print(f"[{region_idx + 1}/{len(REGIONS)}] {name}")
print("Click the view to cycle through span options.")

snx.show(canvas)
zoom_to_fit(view)
snx.run()
