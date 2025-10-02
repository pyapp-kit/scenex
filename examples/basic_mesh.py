import cmap
import numpy as np

import scenex as snx
from scenex.utils.controllers import PanZoomController

try:
    from scenex.imgui import add_imgui_controls
except ImportError:
    print("imgui not available, skipping imgui controls")
    add_imgui_controls = None  # type: ignore[assignment]

vertices = np.array(
    [
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [2, 0, 0],  # 2
        [0, 1, 0],  # 3
        [1, 1, 0],  # 4
        [2, 1, 0],  # 5
        [0, 2, 0],  # 6
        [1, 2, 0],  # 7
        [2, 2, 0],  # 8
    ]
)
faces = np.array(
    [
        [0, 1, 3],
        [1, 2, 5],
        [5, 8, 7],
        [7, 6, 3],
    ]
)

view = snx.View(
    scene=snx.Scene(
        children=[
            snx.Mesh(vertices=vertices, faces=faces, color=cmap.Color("red")),
        ]
    ),
    camera=snx.Camera(controller=PanZoomController(), interactive=True),
)

snx.show(view)
# view.camera.transform = snx.Transform()
view.camera.look_at((1, 1, -1), up=(0, 1, 0))

snx.run()
