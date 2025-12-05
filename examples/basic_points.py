import cmap
import numpy as np

import scenex as snx
from scenex.app.events import Event, MouseMoveEvent
from scenex.utils import projections

# PanZoomController is now part of the main scenex module

# Here is our points data
coords = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
    ]
)
# Scenex provides a few different modes for point sizing.
# "scene" scaling means the point size is defined in world space,
# so it varies as you zoom in and out.
# points = snx.Points(
#     name="point",
#     coords=coords,
#     size=1,  # World-space diameter
#     edge_width=0,
#     scaling="scene"
# )
# "fixed" scaling means the point size is defined in screen space (pixels),
# so it remains constant (fixed) as you zoom in and out.
# You can uncomment the following lines to try it out.
points = snx.Points(
    name="point",
    coords=coords,
    size=20,  # Pixel diameter
    edge_width=0,
    scaling="fixed",
)


# Since ray-point intersections are computed in canvas space, we need view+canvas
view = snx.View(scene=snx.Scene(children=[points]))

view.camera.mouse = snx.PanZoomMouseStrategy()


def _on_view_event(event: Event) -> bool:
    if isinstance(event, MouseMoveEvent):
        intersections = event.world_ray.intersections(view.scene)
        if points in [n for n, _ in intersections]:
            points.face_color = cmap.Color("red")
        else:
            points.face_color = cmap.Color("white")
    return False


view.set_event_filter(_on_view_event)

# Show and position camera
snx.show(view)
view.camera.projection = projections.orthographic(2, 2, 1e5)
view.camera.transform = snx.Transform().translated((0.5, 0.5))
snx.run()
