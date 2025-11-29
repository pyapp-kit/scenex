import numpy as np
import pytest

import scenex as snx
from scenex.app.events import Ray


@pytest.fixture
def volume() -> snx.Volume:
    return snx.Volume(
        data=np.random.randint(0, 255, (60, 100, 100), dtype=np.uint8),
    )


def test_bounding_box(volume: snx.Volume) -> None:
    # Note that the volume has 60 z-slices. But depth comes last in the bounding box!
    exp_bounding_box = np.asarray(((-0.5, -0.5, -0.5), (99.5, 99.5, 59.5)))
    assert np.array_equal(exp_bounding_box, volume.bounding_box)


def test_passes_through(volume: snx.Volume) -> None:
    # Check a ray that passes through the volume hits
    ray = Ray(origin=(50, 50, -1), direction=(0, 0, 1))
    # Note that it intersects at -0.5 because pixel centers are at integer coordinates
    assert volume.passes_through(ray) == 0.5

    # Check a ray that grazes the left edge of the volume hits
    ray = Ray(origin=(-0.5, 0, -1), direction=(0, 0, 1))
    assert volume.passes_through(ray) == 0.5

    # Check a ray that grazes the right edge of the volume misses (the front face)
    ray = Ray(origin=(99.5, 0, -1), direction=(0, 0, 1))
    # Because of symmetry, we miss the front face of the volume but hit the back face
    assert volume.passes_through(ray) == 60.5

    # Check a ray that does not pass through the volume misses
    ray = Ray(origin=(-50, -50, -1), direction=(0, 0, 1))
    assert volume.passes_through(ray) is None
