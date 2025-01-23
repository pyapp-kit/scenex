import numpy as np
from rendercanvas.auto import loop

import scenex as snx

img1 = snx.Image(
    name="Some Image",
    data=np.random.randint(0, 255, (100, 100)).astype(np.uint8),
    clims=(0, 255),
)

img2 = snx.Image(
    data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
    cmap="viridis",
    transform=snx.Transform().scaled((0.7, 0.5)).translated((-10, 20)),
    clims=(0, 255),
)

points = snx.Points(
    coords=np.random.randint(0, 200, (100, 2)).astype(np.uint8),
    size=5,
    face_color="coral",
    edge_color="blue",
    opacity=0.8,
    order=1,
)
scene = snx.Scene(children=[points, img1, img2])
view = snx.View(scene=scene)


snx.show(view)


loop.run()
