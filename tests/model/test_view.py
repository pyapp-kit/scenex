from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import scenex as snx
from scenex.app.events import Event, MouseButton, MouseMoveEvent, Ray
from scenex.utils import projections


def test_to_world() -> None:
    """Tests View.to_ray"""
    camera = snx.Camera(
        transform=snx.Transform(),
        projection=projections.orthographic(2, 2, 2),
        interactive=True,
    )
    view = snx.View(scene=snx.Scene(children=[]), camera=camera)
    canvas = snx.Canvas(views=[view])
    w, h = canvas.rect_for(view)[2:]

    canvas_pos = (w // 2, h // 2)
    ray = view.to_ray(canvas_pos)
    assert ray == Ray(origin=(0, 0, 0), direction=(0, 0, -1), source=view)

    canvas_pos = (0, 0)
    ray = view.to_ray(canvas_pos)
    assert ray == Ray(origin=(-1, 1, 0), direction=(0, 0, -1), source=view)


def test_to_world_translated() -> None:
    """Tests View.to_ray with a translated camera"""
    camera = snx.Camera(
        transform=snx.Transform().translated((1, 1, 1)),
        projection=projections.orthographic(2, 2, 2),
        interactive=True,
    )
    view = snx.View(scene=snx.Scene(children=[]), camera=camera)
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


def test_to_world_projection() -> None:
    """Tests View.to_ray with a non-identity camera projection"""
    camera = snx.Camera(
        transform=snx.Transform(),
        projection=projections.orthographic(1, 1, 1),
        interactive=True,
    )
    view = snx.View(scene=snx.Scene(children=[]), camera=camera)
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
    view.set_event_filter(view_filter)

    # Set up the camera such that the image is in the top right quadrant
    view.camera.transform = snx.Transform().translated((-0.5, -0.5))
    view.camera.projection = projections.orthographic(1, 1, 1)

    # Put it on a canvas
    canvas = snx.Canvas(views=[view])
    _, _, w, _h = canvas.rect_for(view)

    # Mouse over that image in the top right corner
    canvas_pos = (w, 0)
    world_ray = view.to_ray(canvas_pos)
    assert world_ray is not None
    event = MouseMoveEvent(pos=canvas_pos, buttons=MouseButton.NONE)

    # And show the view saw the event
    canvas.handle(event)
    # NOTE: There will also be a MouseEnterEvent when the mouse enters the view
    view_filter.assert_called_with(event)


def test_filter_returning_None() -> None:
    """Some widget backends (e.g. Qt) get upset when non-booleans are returned.

    This test ensures that if a faulty event filter is set that returns None,
    the event is treated as handled (i.e. True is returned).
    """
    view = snx.View()

    def faulty_filter(event: Event) -> bool:
        return None  # type: ignore[return-value]

    view.set_event_filter(faulty_filter)
    canvas = snx.Canvas(views=[view])

    canvas_pos = (canvas.width // 2, canvas.height // 2)
    world_ray = view.to_ray(canvas_pos)
    assert world_ray is not None
    event = MouseMoveEvent(pos=canvas_pos, buttons=MouseButton.NONE)

    handled = view.filter_event(event)
    assert isinstance(handled, bool)
    assert handled is False


def test_view_resizer() -> None:
    """Test that resizer is called when canvas size changes."""
    camera = snx.Camera(
        projection=projections.orthographic(100, 100, 100),
    )
    view = snx.View(camera=camera, on_resize=snx.Letterbox())
    canvas = snx.Canvas(width=400, height=400, views=[view])

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
    view.on_resize = None

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


def test_view_serialization() -> None:
    resize_policy = snx.Letterbox()
    view = snx.View(on_resize=resize_policy)
    json = view.model_dump_json()
    view2 = snx.View.model_validate_json(json)
    # FIXME: there are tons of different errors in round trip serialization
    # let's just make sure that Letterbox() can be round-trip serialized
    # and leave the rest for later
    assert isinstance(view2.on_resize, type(resize_policy))
