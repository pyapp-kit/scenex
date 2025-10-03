from __future__ import annotations

import cmap
import numpy as np
import pytest

import scenex as snx
import scenex.adaptors._vispy as adaptors
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
    """Tests that changing the model changes the view (the Vispy node)."""
    mesh_data = adaptor._vispy_node.mesh_data
    assert mesh_data is not None

    assert np.array_equal(mesh.vertices, np.asarray(mesh_data.get_vertices()))
    mesh.vertices = mesh.vertices * 100
    assert np.array_equal(mesh.vertices, np.asarray(mesh_data.get_vertices()))

    assert np.array_equal(mesh.faces, np.asarray(mesh_data.get_faces()))
    mesh.faces = np.asarray([[0, 2, 3], [3, 1, 0]])
    assert np.array_equal(mesh.faces, np.asarray(mesh_data.get_faces()))

    assert mesh.color is not None
    assert np.array_equal(mesh.color.rgba, adaptor._vispy_node.color.rgba)
    mesh.color = cmap.Color("blue")
    assert np.array_equal(mesh.color.rgba, adaptor._vispy_node.color.rgba)
