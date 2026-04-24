"""
Demonstrates various layout options for positioning a view on a canvas.

Each layout is described by x_start, x_end, y_start, and y_end Dim values.
Click on the image to cycle through the examples. The active layout is printed to the
terminal. Resize the window to see how each configuration responds!
"""

import numpy as np

import scenex as snx
import scenex.app.events as events
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


REGIONS: list[tuple[str, str, str, str, str]] = [
    # Full canvas (default)
    ("Full canvas:", "0%", "100%", "0%", "100%"),
    # Right half, full height
    ("Right half:", "50%", "100%", "0%", "100%"),
    # Fixed 400x400 region at (50, 50)
    ("Fixed 400x400 at (50, 50):", "50px", "450px", "50px", "450px"),
    # Fixed 400x400 region at bottom right corner
    ("Fixed 400x400 at bottom right corner:", "-400px", "100%", "-400px", "100%"),
    # 50px inset on all four sides using negative pixels for far edges
    ("50px inset all sides:", "50px", "-50px", "50px", "-50px"),
    # Centered 200px-wide strip using compound Dim expressions
    (
        "Centered 200px-wide strip (50% - 100px, 50% + 100px):",
        "50% - 100px",
        "50% + 100px",
        "0%",
        "100%",
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


name, *_ = REGIONS[region_idx]
print(f"[{region_idx + 1}/{len(REGIONS)}] {name}")
print("Click the view to cycle through layout options.")

snx.show(canvas)
ci = snx.CanvasInteractor(canvas)
ci.set_view_filter(view, _on_click)
zoom_to_fit(view)
snx.run()
