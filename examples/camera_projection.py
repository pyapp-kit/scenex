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

view.camera.transform = Transform().translated((127.5, 127.5, 300))

# view.camera.projection = projections.orthographic(
#     1.1 * data.shape[0],
#     1.1 *data.shape[1]
# )

near = 300
fov = 2 * atan(data.shape[1] / 2 / 299.5) * 180 / pi
view.camera.projection = projections.perspective(
    fov=fov,
    near=near,
    far=1_000_000,  # Just need something big
)

snx.run()
