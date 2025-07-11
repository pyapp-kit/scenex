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
    # camera=snx.Camera(type="perspective"),
)

# both are optional, just for example
snx.use("pygfx")
# snx.use("vispy")

canvas = snx.show(view)

view.camera.transform = Transform().translated((127.5, 127.5, 228))

# view.camera.projection = projections.orthographic(
#     1.1 * data.shape[0],
#     1.1 *data.shape[1]
# )
view.camera.projection = projections.perspective(
    zoom_factor=0.9,
    fov=70,
    view_width=view.layout.size[0],
    view_height=view.layout.size[1],
    depth=366.9768,
)

snx.run()
