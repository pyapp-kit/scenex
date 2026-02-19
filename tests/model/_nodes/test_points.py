import numpy as np
import pytest

import scenex as snx
from scenex.utils import projections


@pytest.fixture
def points() -> snx.Points:
    return snx.Points(
        vertices=np.random.randint(0, 255, (100, 3), dtype=np.uint8),
    )


def test_bounding_box(points: snx.Points) -> None:
    # This test is a bit tautological, but it does prevent anything crazy from happening
    # :)
    exp_bounding_box = np.asarray(
        (np.min(points.vertices, axis=0), np.max(points.vertices, axis=0))
    )
    assert np.array_equal(exp_bounding_box, points.bounding_box)


def test_points_ray_intersection_screen_space() -> None:
    """Test ray-point intersection in screen/canvas space (fixed scaling).

    Note that ray-point intersections for fixed scaling are computed in canvas space.
    This test is thus more complicated than some other nodes' intersection tests.
    """
    # Create one point with fixed scaling
    vertices = np.array([[1, 1, 0]])
    points = snx.Points(
        vertices=vertices,
        size=4,  # Pixel diameter
        edge_width=0,
        scaling="fixed",
    )

    # Since ray-point intersections are computed in canvas space, we need view+canvas
    view = snx.View(scene=snx.Scene(children=[points]))
    canvas = snx.Canvas(views=[view])

    # Set up camera to look at the points
    view.camera.projection = projections.orthographic(2, 2, 1e5)
    view.camera.transform = snx.Transform().translated((1, 1, 1))
    view.camera.look_at((1, 1, 0), up=(0, 1, 0))

    # Ray going through the center of the point
    canvas_center = (canvas.width // 2, canvas.height // 2)
    ray = canvas.to_world(canvas_center)
    assert ray is not None
    distance = points.passes_through(ray)
    # Should hit the closer point (at distance 1)
    assert distance is not None and np.isclose(distance, 1)

    # Ray going within the point radius should hit
    canvas_offset = (canvas.width // 2 + 1, canvas.height // 2 + 1)
    ray = canvas.to_world(canvas_offset)
    assert ray is not None
    distance = points.passes_through(ray)
    assert distance is not None

    # Ray going outside the point radius should miss
    canvas_far = (canvas.width // 2 + 3, canvas.height // 2 + 5)
    ray = canvas.to_world(canvas_far)
    assert ray is not None
    distance = points.passes_through(ray)
    assert distance is None


def test_points_ray_intersection_world_space() -> None:
    """Test ray-point intersection in world space (scene scaling)."""
    # Create one point with scene scaling
    vertices = np.array([[1, 1, 0]])
    points = snx.Points(
        vertices=vertices,
        size=1,  # World-space diameter
        edge_width=0,
        scaling="scene",
    )

    view = snx.View(scene=snx.Scene(children=[points]))
    canvas = snx.Canvas(views=[view])

    # Set up camera
    view.camera.projection = projections.orthographic(2, 2, 1e5)
    view.camera.transform = snx.Transform().translated((1, 1, 1))
    view.camera.look_at((1, 1, 0), up=(0, 1, 0))

    # Ray going through the center of the points
    canvas_center = (canvas.width // 2, canvas.height // 2)
    ray = canvas.to_world(canvas_center)
    assert ray is not None
    distance = points.passes_through(ray)
    # Should hit the center of the point
    # Since it's a sphere with radius (1/2), the distance should be 0.5
    # As as the camera's at z=1 and the point is at z=0
    assert distance is not None and np.isclose(distance, 0.5)

    # Ray going slightly off center but within radius should hit
    canvas_offset = (canvas.width // 2 + 10, canvas.height // 2)
    ray = canvas.to_world(canvas_offset)
    assert ray is not None
    distance = points.passes_through(ray)
    # The actual intersection distance will vary based on the offset
    # but should be between 0.5 and 1.0
    assert distance is not None and distance > 0.5 and distance < 1.0

    # Ray going far off should miss
    canvas_far = (canvas.width // 4 - 1, canvas.height // 2)
    ray = canvas.to_world(canvas_far)
    assert ray is not None
    distance = points.passes_through(ray)
    assert distance is None


def test_points_ray_intersection_transformed() -> None:
    """Test ray-point intersection with transforms applied."""
    # Simple point at origin
    vertices = np.array([[0, 0, 0]])
    # Transform the point to (1, 1, 0) at the node level
    points = snx.Points(
        vertices=vertices,
        size=4,
        scaling=False,
        transform=snx.Transform().translated((1, 1, 0)),
    )

    view = snx.View(scene=snx.Scene(children=[points]))
    canvas = snx.Canvas(views=[view])

    # Set up camera to look at the transformed point
    view.camera.projection = projections.orthographic(2, 2, 1e5)
    view.camera.transform = snx.Transform().translated((1, 1, 2))
    view.camera.look_at((1, 1, 0), up=(0, 1, 0))

    # Ray going through the center should hit
    canvas_center = (canvas.width // 2, canvas.height // 2)
    ray = canvas.to_world(canvas_center)
    assert ray is not None
    distance = points.passes_through(ray)
    assert distance is not None and np.isclose(distance, 2)

    # Ray going off center should miss
    canvas_offset = (canvas.width // 2 + 10, canvas.height // 2 + 10)
    ray = canvas.to_world(canvas_offset)
    assert ray is not None
    distance = points.passes_through(ray)
    assert distance is None


def test_points_ray_intersection_edge_cases() -> None:
    """Test edge cases for ray-point intersection."""
    # Empty points
    empty_points = snx.Points(
        vertices=np.array([]).reshape(0, 3), size=4, scaling=False
    )
    view = snx.View(scene=snx.Scene(children=[empty_points]))
    canvas = snx.Canvas(views=[view])

    ray = canvas.to_world((canvas.width // 2, canvas.height // 2))
    assert ray is not None
    distance = empty_points.passes_through(ray)
    assert distance is None

    # Points with edge width
    vertices = np.array([[1, 1, 0]])
    points_with_edge = snx.Points(
        vertices=vertices,
        size=2,
        edge_width=2,  # Increases effective radius
        scaling=False,
    )
    # Create a new view with the edge points
    view = snx.View(scene=snx.Scene(children=[points_with_edge]))
    canvas = snx.Canvas(views=[view])

    view.camera.projection = projections.orthographic(2, 2, 1e5)
    view.camera.transform = snx.Transform().translated((1, 1, 2))
    view.camera.look_at((1, 1, 0), up=(0, 1, 0))

    # Should hit due to larger effective radius (size/2 + edge_width = 1 + 2 = 3)
    canvas_offset = (canvas.width // 2 + 2, canvas.height // 2)
    ray = canvas.to_world(canvas_offset)
    assert ray is not None
    distance = points_with_edge.passes_through(ray)
    assert distance is not None
