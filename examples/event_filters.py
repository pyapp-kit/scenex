import cmap
import numpy as np

import scenex as snx
from scenex.app.events import Event, MouseEvent

img = snx.Image(
    data=np.zeros((200, 200)).astype(np.uint8),
    cmap=cmap.Colormap("viridis"),
    clims=(0, 255),
    opacity=0.7,
    interactive=True,
)

view = snx.View(blending="default", scene=snx.Scene(children=[img]))


def _img_filter(event: Event, node: snx.Node) -> bool:
    """Example event drawing a square that reacts to the cursor."""
    # TODO: How might we remove the square when the mouse leaves the image?

    if isinstance(event, MouseEvent) and isinstance(node, snx.Image):
        data = np.zeros((200, 200), dtype=np.uint8)
        x = int(event.world_ray.origin[0])
        min_x = max(0, x - 5)
        max_x = min(data.shape[0], x + 5)

        y = int(event.world_ray.origin[1])
        min_y = max(0, y - 5)
        max_y = min(data.shape[1], y + 5)

        data[min_x:max_x, min_y:max_y] = 255
        node.data = data
    return True


img.set_event_filter(_img_filter)

snx.show(view)
snx.run()
