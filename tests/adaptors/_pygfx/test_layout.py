from __future__ import annotations

import numpy as np
import pytest
from cmap import Color

import scenex as snx
import scenex.adaptors._pygfx as adaptors
from scenex.adaptors import get_adaptor_registry


@pytest.fixture
def view() -> snx.View:
    return snx.View()


@pytest.fixture
def adaptor(view: snx.View) -> adaptors.View:
    adaptor = get_adaptor_registry().get_adaptor(view, create=True)
    assert isinstance(adaptor, adaptors.View)
    return adaptor


def test_background_color(view: snx.View, adaptor: adaptors.View) -> None:
    # Default background color from Layout defaults (black)
    np.testing.assert_array_almost_equal(
        adaptor._background_mat.color_bottom_left.rgba,
        Color("black").rgba,
    )

    # Changing the model color propagates to the native background material
    view.layout.background_color = Color("red")
    np.testing.assert_array_almost_equal(
        adaptor._background_mat.color_bottom_left.rgba,
        Color("red").rgba,
    )

    # Setting to None makes the background transparent
    view.layout.background_color = None
    np.testing.assert_array_almost_equal(
        adaptor._background_mat.color_bottom_left.rgba,
        np.zeros(4),
    )
