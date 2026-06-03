"""Shows how event filters can be used to change the cursor bitmap."""

import numpy as np

import scenex as snx
from scenex.app import CursorType
from scenex.app.events import Event, MouseMoveEvent

# Points data
vertices = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
    ]
)

points = snx.Points(
    name="points",
    vertices=vertices,
    size=18,
    edge_width=0,
    scaling="fixed",
)

view = snx.View(scene=snx.Scene(children=[points]))
view.camera.controller = snx.PanZoom()
canvas = snx.show(view)


def _cursor_filter(event: Event) -> bool:
    if isinstance(event, MouseMoveEvent):
        if not (ray := view.to_ray(event.pos)):
            return False
        if ray.intersections(points):
            snx.set_cursor(canvas, CursorType.CROSS)
        else:
            snx.set_cursor(canvas, CursorType.DEFAULT)
    return False


view.set_event_filter(_cursor_filter)

snx.run()
