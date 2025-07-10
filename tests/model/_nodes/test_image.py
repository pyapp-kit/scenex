import cmap
import numpy as np
import pytest

from scenex.events.events import Ray
from scenex.model._nodes.image import Image


@pytest.fixture
def image() -> Image:
    """Create a simple example image node."""
    return Image(
        data=np.zeros((100, 100), dtype=np.uint8),
        cmap=cmap.Colormap("gray"),
        clims=(0, 255),
        gamma=1.0,
        interpolation="nearest",
    )


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
