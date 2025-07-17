import numpy as np
import pytest

import scenex as snx


@pytest.fixture
def points() -> snx.Points:
    return snx.Points(
        coords=np.random.randint(0, 255, (100, 3), dtype=np.uint8),
    )


def test_bounding_box(points: snx.Points) -> None:
    # This test is a bit tautological, but it does prevent anything crazy from happening
    # :)
    exp_bounding_box = np.asarray(
        (np.min(points.coords, axis=0), np.max(points.coords, axis=0))
    )
    assert np.array_equal(exp_bounding_box, points.bounding_box)
