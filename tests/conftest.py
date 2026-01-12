from collections.abc import Iterator
from unittest.mock import Mock

import cmap
import numpy as np
import pytest

import scenex as snx


@pytest.fixture
def random_points_node() -> snx.Points:
    return snx.Points(
        coords=np.random.randint(0, 200, (100, 2)).astype(np.uint8),
        size=5,
        face_color=cmap.Color("coral"),
        transform=snx.Transform().translated((0, -50)),
    )


@pytest.fixture
def random_image_node() -> snx.Image:
    return snx.Image(
        name="random image",
        data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
        cmap=cmap.Colormap("viridis"),
        transform=snx.Transform().scaled((1.3, 0.5)).translated((-40, 20)),
        clims=(0, 255),
        opacity=0.7,
    )


@pytest.fixture
def random_volume_node() -> snx.Image:
    return snx.Volume(
        name="random volume",
        data=np.random.randint(0, 255, (10, 20, 20)).astype(np.uint8),
        cmap=cmap.Colormap("red"),
        transform=snx.Transform().scaled((-1, -1)).translated((-40, -48)),
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
    random_volume_node: snx.Volume,
    sine_image_node: snx.Image,
) -> snx.Scene:
    return snx.Scene(
        children=[
            sine_image_node,
            random_image_node,
            random_points_node,
            random_volume_node,
        ]
    )


@pytest.fixture
def basic_view(basic_scene: snx.Scene) -> snx.View:
    return snx.View(scene=basic_scene)


@pytest.fixture(autouse=True)
def _doctest_setup(doctest_namespace: dict) -> Iterator[None]:
    """Sets up the doctest namespace.

    The main function currently is to allow examples to call blocking functions (for
    streamlined copy-paste) without actually blocking.
    """
    # Mock snx.run
    snx.run = Mock(return_value=None)

    # TODO: Necessary for the work on https://github.com/pyapp-kit/scenex/pull/42
    # # Mock app().run
    # original_app = app()

    # # Create a wrapper that delegates everything to the real app except run()
    # class MockedApp:
    #     def run(self) -> None:
    #         """No-op run method for doctests."""
    #         pass

    #     def __getattr__(self, name: str) -> Any:
    #         """Delegate all other attributes to the real app."""
    #         return getattr(original_app, name)

    # doctest_namespace["app"] = MockedApp

    yield
