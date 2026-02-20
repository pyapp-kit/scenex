import cmap
import numpy as np
import pytest

import scenex as snx


@pytest.fixture
def text() -> snx.Text:
    return snx.Text(text="Hello, World!", color=cmap.Color("red"), size=12)


def test_bounding_box(text: snx.Text) -> None:
    exp_bounding_box = np.asarray(((-1e-6, -1e-6, -1e-6), (1e-6, 1e-6, 1e-6)))
    assert np.array_equal(exp_bounding_box, text.bounding_box)
