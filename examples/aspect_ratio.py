import cmap
import numpy as np

import scenex as snx
from scenex.utils.controllers import PanZoomController

view = snx.View(
    scene=snx.Scene(
        children=[
            snx.Image(
                data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
                cmap=cmap.Colormap("viridis"),
                transform=snx.Transform().scaled((1.3, 0.5)).translated((-40, 20)),
                clims=(0, 255),
                opacity=0.7,
            ),
        ]
    ),
    camera=snx.Camera(interactive=True),
)
camera_controller = PanZoomController()
camera_controller.maintain_aspect_against(view)
view.camera.set_event_filter(camera_controller)

# both are optional, just for example
# snx.use("pygfx")
# snx.use("vispy")

snx.show(view)
snx.run()
