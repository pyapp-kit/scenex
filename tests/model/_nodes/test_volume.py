import numpy as np
import pytest

import scenex as snx


@pytest.fixture
def volume() -> snx.Volume:
    return snx.Volume(
        data=np.random.randint(0, 255, (60, 100, 100), dtype=np.uint8),
    )


def test_bounding_box(volume: snx.Volume) -> None:
    # Note that the volume has 60 z-slices. But depth comes last in the bounding box!
    exp_bounding_box = np.asarray(((-0.5, -0.5, -0.5), (99.5, 99.5, 59.5)))
    assert np.array_equal(exp_bounding_box, volume.bounding_box)
