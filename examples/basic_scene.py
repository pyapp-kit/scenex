import numpy as np
from rendercanvas.auto import loop

import scenex as snx

# 2d sine wave
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)


view = snx.View(
    scene=snx.Scene(
        children=[
            snx.Image(
                name="Some Image",
                data=Z.astype(np.float32),
                clims=(0, 1),
            ),
            snx.Image(
                data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
                cmap="viridis",
                transform=snx.Transform().scaled((0.7, 0.5)).translated((-10, 20)),
                clims=(0, 255),
                opacity=1,
            ),
            snx.Points(
                coords=np.random.randint(0, 200, (100, 2)).astype(np.uint8),
                size=5,
                face_color="coral",
            ),
        ]
    )
)


snx.show(view)
loop.run()
