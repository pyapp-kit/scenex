import numpy as np

import scenex as snx


def test_basic_scene() -> None:
    # 2d sine wave
    X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    sine_img = (np.sin(X) * np.cos(Y)).astype(np.float32)

    view = snx.View(
        blending="default",
        scene=snx.Scene(
            children=[
                snx.Image(
                    name="sine image",
                    data=sine_img,
                    clims=(-1, 1),
                ),
                snx.Image(
                    data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
                    cmap="viridis",
                    transform=snx.Transform().scaled((1.3, 0.5)).translated((-40, 20)),
                    clims=(0, 255),
                    opacity=0.7,
                ),
                snx.Volume(
                    data=np.random.randint(0, 255, (200, 200, 20)).astype(np.uint8),
                    cmap="viridis",
                    transform=snx.Transform().scaled((1.3, 0.5)).translated((-100, 100)),
                    clims=(0, 255),
                    opacity=0.7,
                ),
                snx.Points(
                    coords=np.random.randint(0, 200, (100, 2)).astype(np.uint8),
                    size=5,
                    face_color="coral",
                    transform=snx.Transform().translated((0, -50)),
                ),
            ]
        ),
    )

    snx.show(view)

test_basic_scene()
