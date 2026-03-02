from __future__ import annotations

import cmap
import numpy as np
import pytest
import vispy.scene

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
        antialias=False,
    )


@pytest.fixture
def adaptor(line: snx.Line) -> adaptors.Line:
    adaptor = get_adaptor_registry().get_adaptor(line, create=True)
    assert isinstance(adaptor, adaptors.Line)
    return adaptor


def test_line_data(line: snx.Line, adaptor: adaptors.Line) -> None:
    """Tests that changing the model vertices updates the Vispy node."""
    assert np.array_equal(line.vertices, np.asarray(adaptor._vispy_node.pos))
    line.vertices = line.vertices * 100
    assert np.array_equal(line.vertices, np.asarray(adaptor._vispy_node.pos))


def test_line_width(line: snx.Line, adaptor: adaptors.Line) -> None:
    assert line.width == adaptor._vispy_node.width
    line.width = 5
    assert adaptor._vispy_node.width == 5


def test_line_color(line: snx.Line, adaptor: adaptors.Line) -> None:
    node = adaptor._vispy_node
    assert isinstance(node, vispy.scene.Line)

    # initial uniform color
    assert isinstance(line.color, snx.UniformColor)
    assert line.color.color.hex == node.color

    # change uniform color
    line.color = snx.UniformColor(color=cmap.Color("blue"))
    assert line.color.color.hex == node.color

    # change to vertex colors
    colors = [
        cmap.Color("green"),
        cmap.Color("yellow"),
        cmap.Color("blue"),
        cmap.Color("red"),
    ]
    line.color = snx.VertexColors(color=colors)
    assert np.array_equal([c.hex for c in line.color.color], node.color)  # pyright: ignore


def test_line_antialias(line: snx.Line, adaptor: adaptors.Line) -> None:
    """Tests that changing the model antialias changes the view (the Vispy node)."""
    node = adaptor._vispy_node
    assert isinstance(node, vispy.scene.Line)
    # Initial state
    assert not line.antialias
    assert node.antialias == line.antialias
    # Change antialias
    line.antialias = True
    assert node.antialias == line.antialias
