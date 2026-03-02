from __future__ import annotations

import cmap
import numpy as np
import pytest

import scenex as snx
import scenex.adaptors._vispy as adaptors
from scenex.adaptors._auto import get_adaptor_registry


@pytest.fixture
def text() -> snx.Text:
    return snx.Text(
        text="Hello, World!",
        color=cmap.Color("red"),
        size=12,
        antialias=False,
    )


@pytest.fixture
def adaptor(text: snx.Text) -> adaptors.Text:
    adaptor = get_adaptor_registry().get_adaptor(text, create=True)
    assert isinstance(adaptor, adaptors.Text)
    return adaptor


def test_data(text: snx.Text, adaptor: adaptors.Text) -> None:
    """Tests that changing the model changes the view (the VisPy node)."""

    assert np.array_equal(text.text, adaptor._vispy_node.text)
    text.text = "Goodbye, World!"
    assert np.array_equal(text.text, adaptor._vispy_node.text)


def test_size(text: snx.Text, adaptor: adaptors.Text) -> None:
    # We don't have a canvas to get the DPI from, so the node assumes it is 96
    # Thus the vispy font size should be 0.75 of the model size (which is in pixels).
    assert np.array_equal(text.size * 0.75, adaptor._vispy_node.font_size)
    text.size = 24
    assert np.array_equal(text.size * 0.75, adaptor._vispy_node.font_size)

    # Now let's set the DPI on the transforms
    adaptor._vispy_node.transforms.dpi = 108  # pyright: ignore[reportOptionalMemberAccess]
    # This is a bit of a hack, but it's the only way to to notify the node that the DPI
    # has been set
    adaptor._vispy_node.transforms.changed()  # pyright: ignore[reportOptionalMemberAccess]
    # Now the font size should be 2/3 of the model size
    assert np.array_equal(text.size * 2 / 3, adaptor._vispy_node.font_size)
    text.size = 12
    assert np.array_equal(text.size * 2 / 3, adaptor._vispy_node.font_size)


def test_color(text: snx.Text, adaptor: adaptors.Text) -> None:
    # NOTE: For some reason vispy TextVisual color is an array of colors
    assert np.array_equal(text.color.rgba, adaptor._vispy_node.color.rgba[0])  # pyright: ignore
    text.color = cmap.Color("blue")
    assert np.array_equal(text.color.rgba, adaptor._vispy_node.color.rgba[0])  # pyright: ignore
