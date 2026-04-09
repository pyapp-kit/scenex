from unittest.mock import MagicMock

import cmap
import numpy as np
import pytest

import scenex as snx
from scenex.app.events import Ray


@pytest.fixture
def image() -> snx.Image:
    return snx.Image(
        data=np.random.randint(0, 255, (200, 100), dtype=np.uint8),
        cmap=cmap.Colormap("gray"),
        clims=(0, 255),
        gamma=1.0,
        interpolation="nearest",
    )


def test_bounding_box(image: snx.Image) -> None:
    """Bounding box sanity test."""
    exp_bounding_box = np.asarray(((-0.5, -0.5, 0), (99.5, 199.5, 0)))
    assert np.array_equal(exp_bounding_box, image.bounding_box)


def test_rgb_bounding_box() -> None:
    """Bounding box sanity test for an RGB image."""
    image = snx.Image(data=np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8))
    exp_bounding_box = np.asarray(((-0.5, -0.5, 0), (99.5, 199.5, 0)))
    assert np.array_equal(exp_bounding_box, image.bounding_box)


def test_passes_through(image: snx.Image) -> None:
    # Check a ray that passes through the image hits
    ray = Ray(origin=(50, 50, 1), direction=(0, 0, -1), source=MagicMock(spec=snx.View))
    assert image.passes_through(ray) == 1

    # Check a ray that grazes the left edge of the image hits
    ray = Ray(
        origin=(-0.5, 0, 1), direction=(0, 0, -1), source=MagicMock(spec=snx.View)
    )
    assert image.passes_through(ray) == 1

    # Check a ray that grazes the right edge of the image misses
    ray = Ray(
        origin=(99.5, 0, 1), direction=(0, 0, -1), source=MagicMock(spec=snx.View)
    )
    assert image.passes_through(ray) is None

    # Check a ray that does not pass through the image misses
    ray = Ray(
        origin=(-50, -50, 1), direction=(0, 0, -1), source=MagicMock(spec=snx.View)
    )
    assert image.passes_through(ray) is None

    # Check a ray that is perpendicular to the image misses
    ray = Ray(origin=(50, 50, 1), direction=(-1, 0, 0), source=MagicMock(spec=snx.View))
    assert image.passes_through(ray) is None
