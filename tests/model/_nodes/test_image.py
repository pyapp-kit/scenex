import numpy as np
import pytest

import scenex as snx


@pytest.fixture
def image() -> snx.Image:
    return snx.Image(
        data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
    )


def test_bounding_box(image: snx.Image) -> None:
    exp_bounding_box = np.asarray(((-0.5, -0.5, 0), (99.5, 99.5, 0)))
    assert np.array_equal(exp_bounding_box, image.bounding_box)
