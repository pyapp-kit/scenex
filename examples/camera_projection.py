from math import atan, pi

import numpy as np

import scenex as snx
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
            ),
        ]
    ),
)

canvas = snx.show(view)

# Translate the camera to the center of the volume, and distance the camera from the
# volume in the z dimension (important for perspective transforms)
view.camera.transform = Transform().translated((127.5, 127.5, 300))

# view.camera.projection = projections.orthographic(
#     1.1 * data.shape[1],
#     1.1 * data.shape[2],
#     1000,
# )

view.camera.projection = projections.perspective(
    # TODO: Create a helper function for this.
    fov=2 * atan(data.shape[1] / 2 / 300) * 180 / pi,
    near=300,
    far=1_000_000,  # Just need something big
)

snx.run()
