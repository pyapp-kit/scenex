"""Demonstrates how event filters can be used to interact with a scene.

In this example, a yellow square is drawn under the mouse cursor when it
hovers over an image. When the mouse leaves the image, a bright border is
drawn around it.
"""

import cmap
import numpy as np

import scenex as snx
from scenex.app.events import Event, MouseEnterEvent, MouseLeaveEvent, MouseMoveEvent

img = snx.Image(
    data=np.zeros((200, 200)).astype(np.uint8),
    cmap=cmap.Colormap("viridis"),
    clims=(0, 255),
    opacity=0.7,
    interactive=True,
)

view = snx.View(scene=snx.Scene(children=[img]))


def _view_filter(event: Event) -> bool:
    """Example event drawing a square that reacts to the cursor."""
    if isinstance(event, MouseMoveEvent):
        if not (ray := view.to_ray(event.pos)):
            return False
        if ray.intersections(img):
            data = np.zeros((200, 200), dtype=np.uint8)
            x = int(ray.origin[0])
            min_x = max(0, x - 5)
            max_x = min(data.shape[0], x + 5)

            y = int(ray.origin[1])
            min_y = max(0, y - 5)
            max_y = min(data.shape[1], y + 5)

            data[min_y:max_y, min_x:max_x] = 255
            img.data = data
        else:
            # Clear the image if the mouse is not over it
            img.data = np.zeros((200, 200), dtype=np.uint8)
            return True
    if isinstance(event, MouseEnterEvent):
        # Restore original colormap and clear the image when mouse enters
        img.data = np.zeros((200, 200), dtype=np.uint8)
    if isinstance(event, MouseLeaveEvent):
        # Add a bright border when mouse leaves the view
        data = np.zeros((200, 200), dtype=np.uint8)
        data[0:3, :] = 255  # Top border
        data[-3:, :] = 255  # Bottom border
        data[:, 0:3] = 255  # Left border
        data[:, -3:] = 255  # Right border
        img.data = data
    return True


canvas = snx.show(view)
ci = snx.CanvasInteractor(canvas)
ci.set_view_filter(view, _view_filter)
snx.run()
