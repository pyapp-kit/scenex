from __future__ import annotations

import cmap
import numpy as np
import pytest

import scenex as snx
import scenex.adaptors._pygfx as adaptors
from scenex.adaptors._auto import get_adaptor_registry


@pytest.fixture
def text() -> snx.Text:
    return snx.Text(
        text="Hello, World!",
        color=cmap.Color("red"),
        size=12,
    )


@pytest.fixture
def adaptor(text: snx.Text) -> adaptors.Text:
    adaptor = get_adaptor_registry().get_adaptor(text, create=True)
    assert isinstance(adaptor, adaptors.Text)
    return adaptor


def test_data(text: snx.Text, adaptor: adaptors.Text) -> None:
    """Tests that changing the model changes the view (the PyGfx node)."""
    mat = adaptor._pygfx_node.material
    assert mat is not None

    assert np.array_equal(text.text, adaptor._pygfx_node._text_blocks[0]._input[1])
    text.text = "Goodbye, World!"
    assert np.array_equal(text.text, adaptor._pygfx_node._text_blocks[0]._input[1])

    assert np.array_equal(text.size, adaptor._pygfx_node.font_size)
    text.size = 24
    assert np.array_equal(text.size, adaptor._pygfx_node.font_size)

    assert text.color is not None
    assert np.array_equal(text.color.rgba, mat.color.rgba)  # pyright: ignore
    text.color = cmap.Color("blue")
    assert np.array_equal(text.color.rgba, mat.color.rgba)  # pyright: ignore
