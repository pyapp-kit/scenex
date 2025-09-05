import cmap
import numpy as np
import pytest

import scenex as snx
from scenex.app.events import Ray
from scenex.model._nodes.image import Image


@pytest.fixture
def image() -> snx.Image:
    return snx.Image(
        data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
        cmap=cmap.Colormap("gray"),
        clims=(0, 255),
        gamma=1.0,
        interpolation="nearest",
    )


def test_bounding_box(image: snx.Image) -> None:
    exp_bounding_box = np.asarray(((-0.5, -0.5, 0), (99.5, 99.5, 0)))
    assert np.array_equal(exp_bounding_box, image.bounding_box)


def test_passes_through(image: Image) -> None:
    # Check a ray that passes through the image
    ray = Ray(origin=(50, 50, 1), direction=(0, 0, -1))
    assert image.passes_through(ray) == 1

    # Check a ray that does not pass through the image
    ray = Ray(origin=(-50, -50, 1), direction=(0, 0, -1))
    assert image.passes_through(ray) is None

    # Check a ray that is perpendicular to the image
    ray = Ray(origin=(50, 50, 1), direction=(-1, 0, 0))
    assert image.passes_through(ray) is None
