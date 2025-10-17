from __future__ import annotations

import cmap
import numpy as np
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
        color=cmap.Color("red"),
        width=1,
    )


@pytest.fixture
def adaptor(line: snx.Line) -> adaptors.Line:
    adaptor = get_adaptor_registry().get_adaptor(line, create=True)
    assert isinstance(adaptor, adaptors.Line)
    return adaptor


def test_data(line: snx.Line, adaptor: adaptors.Line) -> None:
    """Tests that changing the model changes the view (the PyGfx node)."""
    geom = adaptor._pygfx_node.geometry
    mat = adaptor._pygfx_node.material
    assert geom is not None
    assert mat is not None

    assert np.array_equal(line.vertices, geom.positions.data)
    line.vertices = line.vertices * 100
    assert np.array_equal(line.vertices, geom.positions.data)

    assert line.width == mat.thickness  # pyright: ignore
    line.width = 5
    assert line.width == mat.thickness  # pyright: ignore

    assert line.color is not None
    assert np.array_equal(line.color.rgba, mat.color.rgba)  # pyright: ignore
    line.color = cmap.Color("blue")
    assert np.array_equal(line.color.rgba, mat.color.rgba)  # pyright: ignore
