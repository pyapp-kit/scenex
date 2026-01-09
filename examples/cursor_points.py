"""Shows how event filters can be used to change the cursor bitmap."""

import numpy as np

import scenex as snx
from scenex.app import CursorType, app
from scenex.app.events import Event, MouseMoveEvent

# Points data
coords = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
    ]
)

points = snx.Points(
    name="points",
    coords=coords,
    size=18,
    edge_width=0,
    scaling="fixed",
)

view = snx.View(scene=snx.Scene(children=[points]))
view.camera.controller = snx.PanZoom()
canvas = snx.show(view)


def _cursor_filter(event: Event) -> bool:
    if isinstance(event, MouseMoveEvent):
        intersections = event.world_ray.intersections(view.scene)
        if points in [n for n, _ in intersections]:
            app().set_cursor(canvas, CursorType.CROSS)
        else:
            app().set_cursor(canvas, CursorType.DEFAULT)
    return False


view.set_event_filter(_cursor_filter)

snx.run()
