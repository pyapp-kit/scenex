import numpy as np

import scenex as snx
from scenex.events.controllers import OrbitController
from scenex.model._transform import Transform
from scenex.utils import projections

try:
    from imageio.v2 import volread

    url = "https://gitlab.com/scikit-image/data/-/raw/2cdc5ce89b334d28f06a58c9f0ca21aa6992a5ba/cells3d.tif"
    data = np.asarray(volread(url)).astype(np.uint16)[:, 0, :, :]
except ImportError:
    data = np.random.randint(0, 2, (3, 3, 3)).astype(np.uint16)

view = snx.View(
    blending="default",
    scene=snx.Scene(
        children=[
            snx.Volume(
                data=data,
                clims=(data.min(), data.max()),
            )
        ]
    ),
    camera=snx.Camera(
        interactive=True,
    ),
)

# snx.use("vispy")
snx.show(view)

view.camera.transform = (
    Transform()
    .rotated(90, (0, 1, 0))
    .rotated(90, (1, 0, 0))
    .translated((300 + 127.5, 127.5, 29.5))
)
view.camera.projection = projections.perspective(
    fov=70,
    near=1,
    far=1_000_000,  # Just need something big
)
orbit_center = np.mean(view.scene.bounding_box, axis=0)
# orbit_center = (127.5, 127.5, 0)
view.camera.set_event_filter(OrbitController(orbit_center))


snx.run()
