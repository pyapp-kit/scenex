from __future__ import annotations

from unittest.mock import MagicMock, call

import numpy as np
import pytest

import scenex as snx
from scenex.app.events import Event, MouseButton, MouseEnterEvent, MouseMoveEvent, Ray
from scenex.utils import projections


def test_to_ray() -> None:
    """Tests View.to_ray"""
    # Create a single view covering the whole canvas.
    camera = snx.Camera(
        transform=snx.Transform(),
        projection=projections.orthographic(2, 2, 2),
    )
    view = snx.View(scene=snx.Scene(children=[]), camera=camera)
    canvas = snx.Canvas(views=[view])
    w, h = canvas.rect_for(view)[2:]

    # Test center of view/canvas
    canvas_pos = (w // 2, h // 2)
    ray = view.to_ray(canvas_pos)
    assert ray == Ray(origin=(0, 0, 0), direction=(0, 0, -1), source=view)

    # Test top-left corner of view/canvas
    canvas_pos = (0, 0)
    ray = view.to_ray(canvas_pos)
    assert ray == Ray(origin=(-1, 1, 0), direction=(0, 0, -1), source=view)

    # Test past the top-left corner of view/canvas
    # NOTE that view.to_ray still returns a ray even if the canvas position is outside
    # the view's rect - users could check for containment within the view using
    # canvas.content_rect_for(view) if they want.
    canvas_pos = (-w // 2, -h // 2)
    ray = view.to_ray(canvas_pos)
    assert ray == Ray(origin=(-2, 2, 0), direction=(0, 0, -1), source=view)


def test_to_ray_layout() -> None:
    """Tests View.to_ray with a layout"""
    # Create a view with a layout that offsets the content rect by (10, 20)
    camera = snx.Camera(
        transform=snx.Transform(),
        projection=projections.orthographic(2, 2, 2),
    )
    layout = snx.Layout(margin=10)
    view = snx.View(scene=snx.Scene(children=[]), camera=camera, layout=layout)
    canvas = snx.Canvas(views=[view])  # noqa: F841

    ray = view.to_ray((10, 10))
    assert ray == Ray(origin=(-1, 1, 0), direction=(0, 0, -1), source=view)


def test_to_ray_translated() -> None:
    """Tests View.to_ray with a translated camera"""
    # Identity projection, small transformation
    camera = snx.Camera(
        transform=snx.Transform().translated((1, 1, 1)),
        projection=projections.orthographic(2, 2, 2),
    )
    view = snx.View(scene=snx.Scene(children=[]), camera=camera)
    # NOTE: we need a canvas to convert to a ray.
    canvas = snx.Canvas(views=[view])  # noqa: F841

    ray = view.to_ray((0, 0))
    assert ray == Ray(origin=(0, 2, 1), direction=(0, 0, -1), source=view)
    # Rotate counter-clockwise around +Z - we see a clockwise rotation
    # i.e. (-1, 1, 0) moves to the top right corner and (-1, -1, 0) moves to the
    # top left corner
    camera.transform = snx.Transform().rotated(90, (0, 0, 1))
    ray = view.to_ray((0, 0))
    # Rounding errors :(
    assert ray is not None
    assert np.allclose(ray.origin, (-1, -1, 0), atol=1e-7)
    assert np.array_equal(ray.direction, (0, 0, -1))
    assert ray.source == view
    camera.transform = snx.Transform()


def test_to_ray_projection() -> None:
    """Tests View.to_ray with a non-identity camera projection"""
    # Narrowed projection, identity transformation
    camera = snx.Camera(
        transform=snx.Transform(),
        projection=projections.orthographic(1, 1, 1),
    )
    view = snx.View(scene=snx.Scene(children=[]), camera=camera)
    # NOTE: we need a canvas to convert to a ray.
    canvas = snx.Canvas(views=[view])  # noqa: F841

    ray = view.to_ray((0, 0))
    assert ray == Ray(origin=(-0.5, 0.5, 0), direction=(0, 0, -1), source=view)
    camera.projection = snx.Transform()


def test_events() -> None:
    # Create a view with an image
    img = snx.Image(data=np.ones((10, 10), dtype=np.uint8), interactive=True)
    view = snx.View(scene=snx.Scene(children=[img]))
    view_filter = MagicMock()
    view_filter.return_value = False

    # Set up the camera such that the image is in the top right quadrant
    view.camera.transform = snx.Transform().translated((-0.5, -0.5))
    view.camera.projection = projections.orthographic(1, 1, 1)

    # Put it on a canvas
    canvas = snx.Canvas(views=[view])
    ci = snx.CanvasInteractor(canvas)
    ci.set_view_filter(view, view_filter)
    _, _, w, _h = canvas.rect_for(view)

    # Mouse over that image in the top right corner
    canvas_pos = (w, 0)
    world_ray = view.to_ray(canvas_pos)
    assert world_ray is not None
    event = MouseMoveEvent(pos=canvas_pos, buttons=MouseButton.NONE)

    # And show the view saw the event
    ci.handle(event)
    # NOTE that there will also be a MouseEnterEvent
    assert view_filter.call_count == 2
    enter_event = MouseEnterEvent(pos=canvas_pos, buttons=MouseButton.NONE)
    assert view_filter.call_args_list[0] == call(enter_event)
    # And then the MouseMoveEvent we wanted to test
    assert view_filter.call_args_list[1] == call(event)


def test_filter_returning_None() -> None:
    """Some widget backends (e.g. Qt) get upset when non-booleans are returned.

    This test ensures that if a faulty event filter is set that returns None,
    the handle call does not raise and returns a bool.
    """
    view = snx.View()

    def faulty_filter(event: Event) -> bool:
        return None  # type: ignore[return-value]

    canvas = snx.Canvas(views=[view])
    ci = snx.CanvasInteractor(canvas)
    ci.set_view_filter(view, faulty_filter)

    canvas_pos = (canvas.width // 2, canvas.height // 2)
    world_ray = view.to_ray(canvas_pos)
    assert world_ray is not None
    event = MouseMoveEvent(pos=canvas_pos, buttons=MouseButton.NONE)

    handled = ci.handle(event)
    assert isinstance(handled, bool)


def test_view_resizer() -> None:
    """Test that resizer is called when canvas size changes."""
    camera = snx.Camera(
        projection=projections.orthographic(100, 100, 100),
    )
    view = snx.View(camera=camera)
    canvas = snx.Canvas(width=400, height=400, views=[view])
    ci = snx.CanvasInteractor(canvas)
    ci.set_resize_policy(view, snx.Letterbox())

    # Initial aspect should be 1.0 (square)
    # Note that the aspect ratio is stored inversely in the projection matrix,
    # since it maps world space to NDC.
    mat = camera.projection.root
    initial_aspect = abs(mat[1, 1] / mat[0, 0])
    assert initial_aspect == pytest.approx(1.0, rel=1e-6)

    # Resize the canvas to a 2:1 ratio
    canvas.width = 400
    canvas.height = 200

    # Camera projection should now have 2:1 aspect
    mat = camera.projection.root
    new_aspect = abs(mat[1, 1] / mat[0, 0])
    assert new_aspect == pytest.approx(2.0, rel=1e-6)

    # Remove resizer
    ci.set_resize_policy(view, None)

    # Resize canvas again
    canvas.height = 400

    # Projection should remain unchanged
    mat = camera.projection.root
    new_aspect = abs(mat[1, 1] / mat[0, 0])
    assert new_aspect == pytest.approx(2.0, rel=1e-6)


def test_view_canvas_assignment() -> None:
    """Test that assigning a canvas to a view properly updates the view's canvas
    reference and the canvas's views list.
    """
    view = snx.View()
    canvas = snx.Canvas()

    # Assign canvas to view
    view.canvas = canvas

    # Check that the canvas's views list includes the view
    assert view in canvas.views

    # Now set the canvas to None and check that references are cleared
    view.canvas = None
    assert view.canvas is None
    assert view not in canvas.views


def test_letterbox_serialization() -> None:
    """Letterbox can be round-trip serialized."""
    resize_policy = snx.Letterbox()
    json = resize_policy.model_dump_json()
    policy2 = snx.Letterbox.model_validate_json(json)
    assert isinstance(policy2, type(resize_policy))
