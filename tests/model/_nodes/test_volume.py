import numpy as np
import pytest

import scenex as snx


@pytest.fixture
def volume() -> snx.Volume:
    return snx.Volume(
        data=np.random.randint(0, 255, (100, 100, 60), dtype=np.uint8),
    )


def test_bounding_box(volume: snx.Volume) -> None:
    exp_bounding_box = np.asarray(((-0.5, -0.5, -0.5), (99.5, 99.5, 59.5)))
    assert np.array_equal(exp_bounding_box, volume.bounding_box)
