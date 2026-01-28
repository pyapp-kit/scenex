from __future__ import annotations

import cmap
import numpy as np
import pytest

import scenex as snx
import scenex.adaptors._vispy as adaptors
from scenex.adaptors._auto import get_adaptor_registry


@pytest.fixture
def line() -> snx.Line:
    vertices = np.asarray(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [0, 1, 0],  # 2
            [1, 1, 0],  # 3
        ]
    )
    return snx.Line(
        vertices=vertices,
        color=snx.UniformColor(color=cmap.Color("red")),
        width=1,
    )


@pytest.fixture
def adaptor(line: snx.Line) -> adaptors.Line:
    adaptor = get_adaptor_registry().get_adaptor(line, create=True)
    assert isinstance(adaptor, adaptors.Line)
    return adaptor


def test_data(line: snx.Line, adaptor: adaptors.Line) -> None:
    """Tests that changing the model changes the view (the Vispy node)."""
    assert np.array_equal(line.vertices, np.asarray(adaptor._vispy_node.pos))
    line.vertices = line.vertices * 100
    assert np.array_equal(line.vertices, np.asarray(adaptor._vispy_node.pos))

    assert line.width == adaptor._vispy_node.width
    line.width = 5
    assert line.width == adaptor._vispy_node.width

    assert isinstance(line.color, snx.UniformColor)
    assert line.color.color.hex == adaptor._vispy_node.color
    line.color = snx.UniformColor(color=cmap.Color("blue"))
    assert line.color.color.hex == adaptor._vispy_node.color
    line.color = snx.VertexColors(
        color=[
            cmap.Color("green"),
            cmap.Color("yellow"),
            cmap.Color("blue"),
            cmap.Color("red"),
        ],
    )
    assert np.array_equal([c.hex for c in line.color.color], adaptor._vispy_node.color)  # pyright: ignore
