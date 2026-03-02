"""Basic test to verify Line node implementation.

FIXME: Move to tests/model/_nodes/test_line.py
"""

import numpy as np

import scenex as snx
from scenex.utils import projections


def test_line_bounding_box() -> None:
    """Test that bounding box is calculated correctly."""
    vertices = np.array([[0, 0], [2, 1], [1, 2]])
    line = snx.Line(vertices=vertices)

    bbox = line.bounding_box
    expected_min = (0.0, 0.0, 0.0)
    expected_max = (2.0, 2.0, 0.0)

    assert bbox[0] == expected_min
    assert bbox[1] == expected_max


def test_line_ray_intersection() -> None:
    """Test basic ray-line intersection.

    Note that ray-line intersections are computed in canvas space.
    This test is thus more complicated than some other nodes' intersection tests.
    """

    # Simple horizontal line
    vertices = np.array([[0, 1, 1], [2, 1, -1]])
    line = snx.Line(vertices=vertices, width=2)
    # Since ray-line intersections are computed in canvas space, we need view+canvas
    view = snx.View(scene=snx.Scene(children=[line]))
    canvas = snx.Canvas(views=[view])

    # Just barely fit the line into view
    view.camera.transform = projections.orthographic(2, 2, 1e5).translated((1, 1, 1))
    # Camera looking down -Z at the center of the line
    view.camera.look_at((1, 1, 0), up=(0, 1, 0))

    # Ray going through the center of the line
    canvas_center = (canvas.width // 2, canvas.height // 2)
    ray = canvas.to_world(canvas_center)
    assert ray is not None
    distance = line.passes_through(ray)
    assert distance is not None and np.isclose(distance, 1)

    # Ray going through the edge of the line
    canvas_center = (canvas.width, canvas.height // 2)
    ray = canvas.to_world(canvas_center)
    assert ray is not None
    distance = line.passes_through(ray)
    # Note that the distance is larger because the line is at z=-1 at this point
    assert distance is not None and np.isclose(distance, 2)

    # Ray going 1 pixel off center should hit
    canvas_center = (canvas.width // 2, canvas.height // 2 + 1)
    ray = canvas.to_world(canvas_center)
    assert ray is not None
    distance = line.passes_through(ray)
    assert distance is not None and np.isclose(distance, 1)

    # Ray going 2 pixels off center should miss
    canvas_center = (canvas.width // 2, canvas.height // 2 + 2)
    ray = canvas.to_world(canvas_center)
    assert ray is not None
    distance = line.passes_through(ray)
    assert distance is None


def test_line_ray_intersection_transformed() -> None:
    """This test is like test_line_ray_intersection but with transforms."""
    # Simple horizontal line at y=0
    vertices = np.array([[0, 0, 1], [2, 0, -1]])
    # Transformed up by one unit at the node level
    line = snx.Line(
        vertices=vertices, width=2, transform=snx.Transform().translated((0, 1, 0))
    )
    # Since ray-line intersections are computed in canvas space, we need view+canvas
    view = snx.View(
        scene=snx.Scene(
            children=[line],
        )
    )
    canvas = snx.Canvas(views=[view])

    # Just barely fit the line into view
    view.camera.transform = projections.orthographic(2, 2, 1e5).translated((1, 1, 1))
    # Camera looking down -Z at the center of the line
    view.camera.look_at((1, 1, 0), up=(0, 1, 0))

    # Ray going through the center of the line should hit
    canvas_center = (canvas.width // 2, canvas.height // 2)
    ray = canvas.to_world(canvas_center)
    assert ray is not None
    distance = line.passes_through(ray)
    assert distance is not None and np.isclose(distance, 1)

    # Ray going 2 pixels off center should miss
    canvas_center = (canvas.width // 2, canvas.height // 2 + 2)
    ray = canvas.to_world(canvas_center)
    assert ray is not None
    distance = line.passes_through(ray)
    assert distance is None

    # Note that if we transform the scene, the results are the same
    # because the camera is a part of the scene
    view.scene.transform = snx.Transform().translated((0, 1, 0))

    # Ray going through the center of the line should hit
    canvas_center = (canvas.width // 2, canvas.height // 2)
    ray = canvas.to_world(canvas_center)
    assert ray is not None
    distance = line.passes_through(ray)
    assert distance is not None and np.isclose(distance, 1)

    # Ray going 2 pixels off center should miss
    canvas_center = (canvas.width // 2, canvas.height // 2 + 2)
    ray = canvas.to_world(canvas_center)
    assert ray is not None
    distance = line.passes_through(ray)
    assert distance is None
