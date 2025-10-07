from __future__ import annotations

import cmap
import numpy as np
import pytest

import scenex as snx
import scenex.adaptors._pygfx as adaptors
from scenex.adaptors._auto import get_adaptor_registry


@pytest.fixture
def mesh() -> snx.Mesh:
    vertices = np.asarray(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [0, 1, 0],  # 2
            [1, 1, 0],  # 3
        ]
    )
    faces = np.asarray(
        [
            [0, 1, 2],  # 0
            [1, 3, 2],  # 1
        ]
    )
    return snx.Mesh(
        vertices=vertices,
        faces=faces,
        color=cmap.Color("red"),
    )


@pytest.fixture
def adaptor(mesh: snx.Mesh) -> adaptors.Mesh:
    adaptor = get_adaptor_registry().get_adaptor(mesh, create=True)
    assert isinstance(adaptor, adaptors.Mesh)
    return adaptor


def test_data(mesh: snx.Mesh, adaptor: adaptors.Mesh) -> None:
    """Tests that changing the model changes the view (the PyGfx node)."""
    geom = adaptor._pygfx_node.geometry
    mat = adaptor._pygfx_node.material
    assert geom is not None
    assert mat is not None

    assert np.array_equal(mesh.vertices, geom.positions.data)
    mesh.vertices = mesh.vertices * 100
    assert np.array_equal(mesh.vertices, geom.positions.data)

    assert np.array_equal(mesh.faces, geom.indices.data)
    mesh.faces = np.asarray([[0, 2, 3], [3, 1, 0]])
    assert np.array_equal(mesh.faces, geom.indices.data)

    assert mesh.color is not None
    assert np.array_equal(mesh.color.rgba, mat.color.rgba)  # pyright: ignore
    mesh.color = cmap.Color("blue")
    assert np.array_equal(mesh.color.rgba, mat.color.rgba)  # pyright: ignore
