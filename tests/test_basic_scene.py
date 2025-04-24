import numpy as np

import scenex as snx


def test_basic_view(basic_view: snx.View) -> None:
    snx.show(basic_view)
    assert isinstance(basic_view.model_dump(), dict)
