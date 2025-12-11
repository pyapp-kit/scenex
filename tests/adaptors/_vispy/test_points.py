from __future__ import annotations

import cmap
import numpy as np
import pytest
import vispy.scene

import scenex as snx
import scenex.adaptors._vispy as adaptors
from scenex.adaptors._auto import get_adaptor_registry


@pytest.fixture
def points() -> snx.Points:
    coords = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]])
    return snx.Points(
        coords=coords,
        size=10,
        face_color=snx.ColorModel(type="uniform", color=cmap.Color("red")),
        edge_color=snx.ColorModel(type="uniform", color=cmap.Color("white")),
        edge_width=1.0,
        symbol="disc",
        scaling=False,
        antialias=0,
    )


@pytest.fixture
def adaptor(points: snx.Points) -> adaptors.Points:
    adaptor = get_adaptor_registry().get_adaptor(points, create=True)
    assert isinstance(adaptor, adaptors.Points)
    return adaptor


def test_points_data(points: snx.Points, adaptor: adaptors.Points) -> None:
    """Tests that changing the model data changes the view (the Vispy node)."""
    node = adaptor._vispy_node
    assert isinstance(node, vispy.scene.Markers)
    # Initial state
    assert np.array_equal(node._data["a_position"], points.coords)  # pyright: ignore
    # Change coords
    new_coords = np.array([[10, 10, 0], [20, 20, 0]])
    points.coords = new_coords
    assert np.array_equal(node._data["a_position"], new_coords)  # pyright: ignore


def test_points_size(points: snx.Points, adaptor: adaptors.Points) -> None:
    """Tests that changing the model size changes the view (the Vispy node)."""
    node = adaptor._vispy_node
    assert isinstance(node, vispy.scene.Markers)
    # Initial state
    assert np.array_equal(
        node._data["a_size"],  # pyright: ignore
        points.size * np.ones(len(points.coords)),  # pyright: ignore
    )
    # Change size
    points.size = 5
    assert np.array_equal(node._data["a_size"], 5 * np.ones(len(points.coords)))  # pyright: ignore


def test_points_opacity(points: snx.Points, adaptor: adaptors.Points) -> None:
    """Tests that changing the model opacity changes the view (the Vispy node)."""
    node = adaptor._vispy_node
    assert isinstance(node, vispy.scene.Markers)
    # Initial state
    assert node.alpha == points.opacity
    # Change opacity
    points.opacity = 0.5
    assert node.alpha == 0.5


def test_points_antialias(points: snx.Points, adaptor: adaptors.Points) -> None:
    """Tests that changing the model antialias changes the view (the Vispy node)."""
    node = adaptor._vispy_node
    assert isinstance(node, vispy.scene.Markers)
    # Initial state
    assert points.antialias == 0
    assert node.antialias == points.antialias
    # Change antialias
    points.antialias = 1
    assert node.antialias == 1


def test_points_scaling(points: snx.Points, adaptor: adaptors.Points) -> None:
    """Tests that changing the model scaling changes the view (the Vispy node)."""
    node = adaptor._vispy_node
    assert isinstance(node, vispy.scene.Markers)
    # Initial state
    assert not points.scaling
    assert node.scaling == "fixed"
    # Change scaling
    points.scaling = True  # "scene"
    assert node.scaling == "scene"
    points.scaling = "visual"
    assert node.scaling == "visual"
    points.scaling = "scene"
    assert node.scaling == "scene"
    points.scaling = "fixed"
    assert node.scaling == "fixed"


def test_points_color(points: snx.Points, adaptor: adaptors.Points) -> None:
    """Test color updates for Points."""
    node = adaptor._vispy_node

    # Initial uniform colors
    assert points.face_color.color == cmap.Color("red")
    for face in node._data["a_bg_color"]:  # pyright: ignore
        np.testing.assert_allclose(face, points.face_color.color)
    assert points.edge_color.color == cmap.Color("white")
    for face in node._data["a_fg_color"]:  # pyright: ignore
        np.testing.assert_allclose(face, points.edge_color.color)

    # Change face color to uniform Blue
    points.face_color = snx.ColorModel(type="uniform", color=cmap.Color("blue"))
    for face in node._data["a_bg_color"]:  # pyright: ignore
        np.testing.assert_allclose(face, points.face_color.color)

    # Change edge color to uniform Green
    points.edge_color = snx.ColorModel(type="uniform", color=cmap.Color("green"))
    for face in node._data["a_fg_color"]:  # pyright: ignore
        np.testing.assert_allclose(face, points.edge_color.color)

    # Change face color to Vertex
    colors = [cmap.Color("red"), cmap.Color("green"), cmap.Color("blue")]
    points.face_color = snx.ColorModel(type="vertex", color=colors)
    for actual, expected in zip(node._data["a_bg_color"], colors, strict=False):  # pyright: ignore
        np.testing.assert_allclose(actual, expected)

    # Change edge color to Vertex
    edge_colors = [cmap.Color("yellow"), cmap.Color("cyan"), cmap.Color("magenta")]
    points.edge_color = snx.ColorModel(type="vertex", color=edge_colors)
    for actual, expected in zip(node._data["a_fg_color"], edge_colors, strict=False):  # pyright: ignore
        np.testing.assert_allclose(actual, expected)
