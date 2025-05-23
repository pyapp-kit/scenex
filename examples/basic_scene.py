import cmap
import numpy as np

import scenex as snx

view = snx.View(
    blending="default",
    scene=snx.Scene(
        children=[
            snx.Image(
                data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
                cmap=cmap.Colormap("viridis"),
                transform=snx.Transform().scaled((1.3, 0.5)).translated((-40, 20)),
                clims=(0, 255),
                opacity=0.7,
            ),
            snx.Points(
                coords=np.random.randint(0, 200, (100, 2)).astype(np.uint8),
                size=5,
                face_color=cmap.Color("coral"),
                edge_color=cmap.Color("purple"),
                transform=snx.Transform().translated((0, -50)),
            ),
        ]
    ),
)

# example of adding an object to a scene
X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
sine_img = (np.sin(X) * np.cos(Y)).astype(np.float32)
image = snx.Image(name="sine image", data=sine_img, clims=(-1, 1))
view.scene.add_child(image)

snx.show(view)
snx.loop()
