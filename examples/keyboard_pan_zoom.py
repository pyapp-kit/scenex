"""Demonstrates keyboard-driven pan and zoom on top of the PanZoom controller.

Arrow keys pan the view; + and - zoom in and out. Mouse drag and scroll
continue to work as normal via the PanZoom controller.
"""

import numpy as np
from app_model.types import KeyCode

import scenex as snx
from scenex.app.events import Event, KeyPressEvent

# Build a recognisable test image: a grid of bright dots on a dark background.
rng = np.random.default_rng(0)
data = np.zeros((512, 512), dtype=np.uint8)
coords = rng.integers(8, 504, size=(200, 2))
for y, x in coords:
    data[y - 3 : y + 3, x - 3 : x + 3] = 255

view = snx.View(
    scene=snx.Scene(children=[snx.Image(data=data)]),
    camera=snx.Camera(controller=snx.PanZoom(), interactive=True),
)
canvas = snx.Canvas(views=[view])

_PAN_STEP = 20.0  # world units per arrow-key press
_ZOOM_STEP = 1.25  # multiplicative factor per +/- press


def _key_filter(event: Event) -> bool:
    """Pan with arrow keys; zoom with + / -."""
    if not isinstance(event, KeyPressEvent):
        return False

    key = event.key  # KeyCode (or KeyCombo for modified keys)
    cam = view.camera

    if key == KeyCode.UpArrow:
        cam.transform = cam.transform.translated((0, _PAN_STEP))
    elif key == KeyCode.DownArrow:
        cam.transform = cam.transform.translated((0, -_PAN_STEP))
    elif key == KeyCode.LeftArrow:
        cam.transform = cam.transform.translated((-_PAN_STEP, 0))
    elif key == KeyCode.RightArrow:
        cam.transform = cam.transform.translated((_PAN_STEP, 0))
    elif key in (KeyCode.Equal, KeyCode.NumpadAdd):  # + / numpad +
        s = _ZOOM_STEP
        cam.projection = cam.projection.scaled((s, s, 1.0))
    elif key == KeyCode.Minus:  # -
        s = 1.0 / _ZOOM_STEP
        cam.projection = cam.projection.scaled((s, s, 1.0))
    else:
        print(f"Unhandled key: {key}")
        return False

    return True


canvas.set_event_filter(_key_filter)

snx.show(canvas)
snx.run()
