import numpy as np
import pytest
import scenex as snx


@pytest.fixture
def random_points_node() -> snx.Points:
    return snx.Points(
        coords=np.random.randint(0, 200, (100, 2)).astype(np.uint8),
        size=5,
        face_color="coral",
        transform=snx.Transform().translated((0, -50)),
    )


@pytest.fixture
def random_image_node() -> snx.Image:
    return snx.Image(
        name="random image",
        data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
        cmap="viridis",
        transform=snx.Transform().scaled((1.3, 0.5)).translated((-40, 20)),
        clims=(0, 255),
        opacity=0.7,
    )


@pytest.fixture
def sine_image_node() -> snx.Image:
    # 2d sine wave
    X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    sine_img = (np.sin(X) * np.cos(Y)).astype(np.float32)
    return snx.Image(name="sine image", data=sine_img, clims=(-1, 1))


@pytest.fixture
def basic_scene(
    random_points_node: snx.Points,
    random_image_node: snx.Image,
    sine_image_node: snx.Image,
) -> snx.Scene:
    return snx.Scene(children=[sine_image_node, random_image_node, random_points_node])


@pytest.fixture
def basic_view(basic_scene: snx.Scene) -> snx.View:
    return snx.View(blending="default", scene=basic_scene)
