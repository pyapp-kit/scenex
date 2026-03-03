"""
Demonstrates the different region types for positioning a view on a canvas.

Click anywhere on the view to cycle through region options. The active
region is printed to the terminal. Resize the window to see how each
region responds.
"""

import numpy as np

import scenex as snx
import scenex.app.events as events
from scenex import FractionalRegion, PixelRegion
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


REGIONS: list[tuple[str, PixelRegion | FractionalRegion]] = [
    # --- FractionalRegion: scalar (same fraction for both axes) ---
    (
        "FractionalRegion start=0 end=1 total=1 (scalar): entire canvas",
        FractionalRegion(start=0, end=1, total=1),
    ),
    (
        "FractionalRegion start=0 end=1 total=2 (scalar): top-left quarter",
        FractionalRegion(start=0, end=1, total=2),
    ),
    (
        "FractionalRegion start=1 end=2 total=3 (scalar): middle ninth",
        FractionalRegion(start=1, end=2, total=3),
    ),
    # --- FractionalRegion: tuple (independent x/y fractions) ---
    (
        "FractionalRegion start=(0,0) end=(1,1) total=(2,1): left half, full height",
        FractionalRegion(start=(0, 0), end=(1, 1), total=(2, 1)),
    ),
    (
        "FractionalRegion start=(1,0) end=(2,1) total=(2,1): right half, full height",
        FractionalRegion(start=(1, 0), end=(2, 1), total=(2, 1)),
    ),
    # --- PixelRegion: absolute ---
    (
        "PixelRegion left=50 top=50 width=400 height=400: fixed 400x400 at (50,50)",
        PixelRegion(left=50, top=50, width=400, height=400),
    ),
    (
        "PixelRegion left=50 top=50: stretches to fill right and bottom",
        PixelRegion(left=50, top=50),
    ),
    # --- PixelRegion: negative (from far edge) ---
    (
        "PixelRegion left=-400 top=50 width=400 height=400: pin to the right edge",
        PixelRegion(left=-400, top=50, width=400, height=400),
    ),
    (
        "PixelRegion left=50 right=-50 top=50 bottom=-50: 50px margin all sides",
        PixelRegion(left=50, right=-50, top=50, bottom=-50),
    ),
    (
        "PixelRegion left=0 top=-400 width=400 height=400: pins to the bottom",
        PixelRegion(left=100, top=-400, width=400, height=400),
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
        name, region = REGIONS[region_idx]
        view.layout.region = region
        print(f"[{region_idx + 1}/{len(REGIONS)}] {name}")
    return False


view.set_event_filter(_on_click)

name, _ = REGIONS[region_idx]
print(f"[{region_idx + 1}/{len(REGIONS)}] {name}")
print("Click the view to cycle through region options.")

snx.show(canvas)
zoom_to_fit(view)
snx.run()
