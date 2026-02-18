from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import scenex as snx
from scenex.app.events import Event, MouseButton, MouseMoveEvent
from scenex.utils import projections


def test_events() -> None:
    # Create a view with an image
    img = snx.Image(data=np.ones((10, 10), dtype=np.uint8), interactive=True)
    view = snx.View(scene=snx.Scene(children=[img]))
    view_filter = MagicMock()
    view_filter.return_value = False
    view.set_event_filter(view_filter)

    # Set up the camera
    # Such that the image is in the top right quadrant
    view.camera.transform = snx.Transform().translated((-0.5, -0.5))
    view.camera.projection = projections.orthographic(1, 1, 1)

    # Put it on a canvas
    canvas = snx.Canvas(
        width=int(view.layout.width), height=int(view.layout.height), views=[view]
    )

    # Mouse over that image in the top right corner
    canvas_pos = (view.layout.width, 0)
    world_ray = canvas.to_world(canvas_pos)
    assert world_ray is not None
    event = MouseMoveEvent(
        canvas_pos=canvas_pos, world_ray=world_ray, buttons=MouseButton.NONE
    )

    # And show the view saw the event
    canvas.handle(event)
    view_filter.assert_called_once_with(event)


def test_filter_returning_None() -> None:
    """Some widget backends (e.g. Qt) get upset when non-booleans are returned.

    This test ensures that if a faulty event filter is set that returns None,
    the event is treated as handled (i.e. True is returned).
    """
    # Create a view...
    view = snx.View()

    # ...with a faulty event filter...
    def faulty_filter(event: Event) -> bool:
        return None  # type: ignore[return-value]

    view.set_event_filter(faulty_filter)

    # ...put it on a canvas...
    canvas = snx.Canvas(
        width=int(view.layout.width), height=int(view.layout.height), views=[view]
    )

    # ...and create a mock event...
    canvas_pos = (0, 0)
    world_ray = canvas.to_world(canvas_pos)
    assert world_ray is not None
    event = MouseMoveEvent(
        canvas_pos=canvas_pos, world_ray=world_ray, buttons=MouseButton.NONE
    )

    # ...to test handling...
    handled = view.filter_event(event)
    assert isinstance(handled, bool)
    assert handled is False


def test_view_resizer() -> None:
    """Test that resizer is called when layout size changes."""
    camera = snx.Camera(
        projection=projections.orthographic(100, 100, 100),
    )
    view = snx.View(camera=camera, on_resize=snx.Letterbox())

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

    # Remove resizer
    view.on_resize = None

    # Change layout size again
    view.layout.height = 400

    # Projection should remain unchanged
    mat = camera.projection.root
    new_aspect = abs(mat[1, 1] / mat[0, 0])
    assert new_aspect == pytest.approx(2.0, rel=1e-6)


def test_view_serialization() -> None:
    resize_policy = snx.Letterbox()
    view = snx.View(on_resize=resize_policy)
    json = view.model_dump_json()
    view2 = snx.View.model_validate_json(json)
    # FIXME: there are tons of different errors in round trip serialization
    # let's just make sure that Letterbox() can be round-trip serialized
    # and leave the rest for later
    assert isinstance(view2.on_resize, type(resize_policy))
