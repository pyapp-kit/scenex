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
        color=snx.ColorModel(type="uniform", color=cmap.Color("red")),
    )


@pytest.fixture
def adaptor(mesh: snx.Mesh) -> adaptors.Mesh:
    adaptor = get_adaptor_registry().get_adaptor(mesh, create=True)
    assert isinstance(adaptor, adaptors.Mesh)
    return adaptor


def test_data(mesh: snx.Mesh, adaptor: adaptors.Mesh) -> None:
    """Tests that changing the model changes the view (the Vispy node)."""
    # Change vertices
    assert np.array_equal(
        mesh.vertices,
        np.asarray(adaptor._vispy_node.mesh_data.get_vertices()),  # pyright: ignore
    )
    mesh.vertices = mesh.vertices * 100
    assert np.array_equal(
        mesh.vertices,
        np.asarray(adaptor._vispy_node.mesh_data.get_vertices()),  # pyright: ignore
    )

    # Change faces
    assert np.array_equal(
        mesh.vertices,
        np.asarray(adaptor._vispy_node.mesh_data.get_vertices()),  # pyright: ignore
    )
    mesh.faces = np.asarray([[0, 2, 3], [3, 1, 0]])
    assert np.array_equal(
        mesh.vertices,
        np.asarray(adaptor._vispy_node.mesh_data.get_vertices()),  # pyright: ignore
    )


def test_color(mesh: snx.Mesh, adaptor: adaptors.Mesh) -> None:
    """Tests that changing the model color changes the view (the Vispy node)."""
    # Change color
    assert mesh.color is not None
    assert np.array_equal(mesh.color.color.rgba, adaptor._vispy_node.color.rgba)  # type: ignore
    mesh.color = snx.ColorModel(type="uniform", color=cmap.Color("blue"))
    assert np.array_equal(mesh.color.color.rgba, adaptor._vispy_node.color.rgba)  # type: ignore

    # Change to vertex colors
    mesh_data = adaptor._vispy_node.mesh_data
    assert mesh_data is not None
    stored_colors = mesh_data.get_vertex_colors()
    assert stored_colors is None

    colors = [
        cmap.Color("red"),
        cmap.Color("green"),
        cmap.Color("blue"),
        cmap.Color("yellow"),
    ]
    mesh.color = snx.ColorModel(type="vertex", color=colors)
    # VisPy stores vertex colors in mesh_data
    # Note: VisPy might return RGBA or just RGB depending on input, but usually RGBA
    # We check if the stored colors match our input
    stored_colors = mesh_data.get_vertex_colors()
    assert stored_colors is not None
    expected_colors = np.array([c.rgba for c in colors])
    np.testing.assert_allclose(stored_colors, expected_colors)
    # TODO: Support face colors
