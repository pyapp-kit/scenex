"""Tests for View + ResizeStrategy integration."""

import pytest

from scenex.model import Camera, Letterbox, View
from scenex.utils import projections


def test_view_resizer_integration() -> None:
    """Test that resizer is called when layout size changes."""
    camera = Camera(
        projection=projections.orthographic(100, 100, 100),
    )
    view = View(camera=camera, resize=Letterbox())

    # Initial aspect should be 1.0 (square)
    # Note that the aspect ratio is stored inversely in the projection matrix,
    # since it maps world space to NDC.
    mat = camera.projection.root
    initial_aspect = abs(mat[1, 1] / mat[0, 0])
    assert initial_aspect == pytest.approx(1.0, rel=1e-6)

    # Change layout size
    view.layout.width = 400
    view.layout.height = 200

    # Camera projection should now have 2:1 aspect
    mat = camera.projection.root
    new_aspect = abs(mat[1, 1] / mat[0, 0])
    assert new_aspect == pytest.approx(2.0, rel=1e-6)


def test_view_resizer_removed() -> None:
    """Test that removing resizer disconnects the callback."""
    camera = Camera(projection=projections.orthographic(100, 100, 100))
    view = View(camera=camera, resize=Letterbox())

    # Remove resizer
    view.resize = None

    # Get initial projection
    initial_projection = camera.projection.root.copy()

    # Change layout size
    view.layout.width = 400
    view.layout.height = 200

    # Projection should remain unchanged (no resizer)
    assert (camera.projection.root == initial_projection).all()


def test_view_resizer_multiple_size_changes() -> None:
    """Test that resizer responds to multiple size changes."""
    camera = Camera(projection=projections.orthographic(100, 100, 100))
    view = View(camera=camera, resize=Letterbox())

    # First resize: 2:1
    view.layout.width = 400
    view.layout.height = 200
    mat = camera.projection.root
    aspect1 = abs(mat[1, 1] / mat[0, 0])
    assert aspect1 == pytest.approx(2.0, rel=1e-6)

    # Second resize: 1:2
    view.layout.width = 200
    view.layout.height = 400
    mat = camera.projection.root
    aspect2 = abs(mat[1, 1] / mat[0, 0])
    assert aspect2 == pytest.approx(0.5, rel=1e-6)

    # Third resize: 1:1
    view.layout.width = 300
    view.layout.height = 300
    mat = camera.projection.root
    aspect3 = abs(mat[1, 1] / mat[0, 0])
    assert aspect3 == pytest.approx(1.0, rel=1e-6)
