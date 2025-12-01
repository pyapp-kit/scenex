import cmap
import numpy as np

import scenex as snx

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
    camera=snx.Camera(controller=snx.PanZoomController(), interactive=True),
)

# TODO: The new declarative PanZoomController doesn't yet support maintaining aspect.
# This example is temporarily disabled until that feature is implemented.
# The old implementation used imperative event listeners to adjust projection on resize.

# The maintain_aspect_against feature needs to be re-implemented
# in the new declarative controller system.
# Old code:
# camera_controller = PanZoomController()
# camera_controller.maintain_aspect_against(view)
# view.camera.set_event_filter(camera_controller)

snx.show(view)
snx.run()
