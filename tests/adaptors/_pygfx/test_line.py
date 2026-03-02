from __future__ import annotations

import cmap
import numpy as np
import pygfx
import pytest

import scenex as snx
import scenex.adaptors._pygfx as adaptors
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
    """Tests that changing the model vertices updates the PyGfx node."""
    geom = adaptor._pygfx_node.geometry
    assert geom is not None

    assert np.array_equal(line.vertices, geom.positions.data)
    line.vertices = line.vertices * 100
    assert np.array_equal(line.vertices, geom.positions.data)


def test_line_width(line: snx.Line, adaptor: adaptors.Line) -> None:
    mat = adaptor._pygfx_node.material
    assert isinstance(mat, pygfx.LineMaterial)

    assert line.width == mat.thickness
    line.width = 5
    assert mat.thickness == 5


def test_line_color(line: snx.Line, adaptor: adaptors.Line) -> None:
    mat = adaptor._pygfx_node.material
    assert isinstance(mat, pygfx.LineMaterial)
    geom = adaptor._pygfx_node.geometry

    # initial uniform color
    assert np.array_equal(line.color.color.rgba, mat.color.rgba)  # type: ignore

    # change uniform color
    line.color = snx.UniformColor(color=cmap.Color("blue"))
    assert np.array_equal(line.color.color.rgba, mat.color.rgba)

    # change to vertex colors
    colors = [
        cmap.Color("green"),
        cmap.Color("yellow"),
        cmap.Color("blue"),
        cmap.Color("red"),
    ]
    line.color = snx.VertexColors(color=colors)
    assert np.array_equal(
        np.asarray([c.rgba for c in line.color.color], dtype=np.float32),
        geom.colors.data,  # pyright: ignore
    )


def test_line_antialias(line: snx.Line, adaptor: adaptors.Line) -> None:
    node = adaptor._pygfx_node
    mat = node.material
    assert isinstance(mat, pygfx.LineMaterial)

    # Initial state
    assert not line.antialias
    assert mat.aa == line.antialias
    # Change antialias
    line.antialias = True
    assert mat.aa
