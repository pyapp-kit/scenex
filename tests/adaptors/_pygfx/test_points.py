from __future__ import annotations

import cmap
import numpy as np
import pygfx
import pytest

import scenex as snx
import scenex.adaptors._pygfx as adaptors
from scenex.adaptors._auto import get_adaptor_registry


@pytest.fixture
def points() -> snx.Points:
    vertices = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]])
    return snx.Points(
        vertices=vertices,
        size=10,
        face_color=snx.UniformColor(color=cmap.Color("red")),
        edge_color=snx.UniformColor(color=cmap.Color("white")),
        edge_width=1.0,
        symbol="disc",
        scaling="fixed",
        antialias=False,
    )


@pytest.fixture
def adaptor(points: snx.Points) -> adaptors.Points:
    adaptor = get_adaptor_registry().get_adaptor(points, create=True)
    assert isinstance(adaptor, adaptors.Points)
    return adaptor


def test_points_data(points: snx.Points, adaptor: adaptors.Points) -> None:
    """Tests that changing the model data updates the PyGFX node."""
    node = adaptor._pygfx_node
    assert isinstance(node, pygfx.Points)
    geom = node.geometry
    assert isinstance(geom, pygfx.Geometry)

    # initial data
    assert np.array_equal(geom.positions.data, points.vertices)

    # update vertices
    new_vertices = np.array([[5, 5, 0], [6, 6, 0]])
    points.vertices = new_vertices
    assert np.array_equal(geom.positions.data, new_vertices)


def test_points_size(points: snx.Points, adaptor: adaptors.Points) -> None:
    node = adaptor._pygfx_node
    mat = node.material
    assert isinstance(mat, pygfx.PointsMaterial)

    assert mat.size == points.size
    points.size = 5
    assert mat.size == 5


def test_points_opacity(points: snx.Points, adaptor: adaptors.Points) -> None:
    node = adaptor._pygfx_node
    mat = node.material
    assert isinstance(mat, pygfx.PointsMaterial)

    assert mat.opacity == points.opacity
    points.opacity = 0.5
    assert mat.opacity == 0.5


def test_points_antialias(points: snx.Points, adaptor: adaptors.Points) -> None:
    node = adaptor._pygfx_node
    mat = node.material
    assert isinstance(mat, pygfx.PointsMaterial)

    # Initial state
    assert not points.antialias
    assert mat.aa == points.antialias
    # Change antialias
    points.antialias = True
    assert mat.aa


def test_points_scaling(points: snx.Points, adaptor: adaptors.Points) -> None:
    node = adaptor._pygfx_node
    mat = node.material
    assert isinstance(mat, pygfx.PointsMaterial)

    # "visual" -> model,
    points.scaling = "visual"
    assert mat.size_space == "model"
    # "scene" -> world
    points.scaling = "scene"
    assert mat.size_space == "world"
    # "fixed" -> screen
    points.scaling = "fixed"
    assert mat.size_space == "screen"


def test_points_color(points: snx.Points, adaptor: adaptors.Points) -> None:
    node = adaptor._pygfx_node
    mat = node.material
    assert isinstance(mat, pygfx.PointsMarkerMaterial)
    geom = node.geometry
    assert isinstance(geom, pygfx.Geometry)

    # initial uniform colors
    assert mat.color_mode == "uniform"
    np.testing.assert_allclose(mat.color, points.face_color.color.rgba)  # type: ignore
    assert mat.edge_color_mode == "uniform"
    np.testing.assert_allclose(mat.edge_color, points.edge_color.color.rgba)  # type: ignore

    # change face uniform
    points.face_color = snx.UniformColor(color=cmap.Color("blue"))
    assert mat.color_mode == "uniform"
    np.testing.assert_allclose(mat.color, points.face_color.color.rgba)  # pyright: ignore

    # change edge uniform
    points.edge_color = snx.UniformColor(color=cmap.Color("green"))
    assert mat.edge_color_mode == "uniform"
    np.testing.assert_allclose(mat.edge_color, points.edge_color.color.rgba)  # pyright: ignore

    # change face vertex
    colors = [cmap.Color("red"), cmap.Color("green"), cmap.Color("blue")]
    points.face_color = snx.VertexColors(color=colors)
    assert mat.color_mode == "vertex"
    np.testing.assert_allclose(
        geom.colors.data, np.array([c.rgba for c in colors], dtype=np.float32)
    )

    # change edge vertex
    edge_colors = [cmap.Color("yellow"), cmap.Color("cyan"), cmap.Color("magenta")]
    points.edge_color = snx.VertexColors(color=edge_colors)
    assert mat.edge_color_mode == "vertex"
    np.testing.assert_allclose(
        geom.edge_colors.data, np.array([c.rgba for c in edge_colors], dtype=np.float32)
    )
