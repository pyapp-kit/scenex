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

snx.use("vispy")
snx.show(view)

# Orbit around the center of the volume
orbit_center = np.mean(np.asarray(view.scene.bounding_box), axis=0)

# Place the camera along the x axis, looking at the orbit center
# TODO: Need a look at method
view.camera.transform = (
    Transform()
    .rotated(90, (0, 1, 0))
    .rotated(90, (1, 0, 0))
    .translated(orbit_center)
    .translated((300, 0, 0))
)
# Perspective projection for 3D
view.camera.projection = projections.perspective(
    fov=70,
    near=1,
    far=1_000_000,  # Just need something big
)
view.camera.set_event_filter(OrbitController(orbit_center))


snx.run()
