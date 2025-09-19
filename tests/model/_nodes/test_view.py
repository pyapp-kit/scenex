from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

import scenex as snx
from scenex.app.events import MouseButton, MouseMoveEvent
from scenex.utils import projections


def test_events() -> None:
    # Create a view with an image
    img = snx.Image(data=np.ones((10, 10), dtype=np.uint8), interactive=True)
    img_filter = MagicMock()
    img.set_event_filter(img_filter)

    view = snx.View(scene=snx.Scene(children=[img]))
    view_filter = MagicMock()
    view_filter.return_value = False
    view.set_event_filter(view_filter)

    # Set up the camera
    # Such that the image is in the top right quadrant
    view.camera.transform = snx.Transform().translated((-0.5, -0.5))
    view.camera.projection = projections.orthographic(1, 1, 1)

    # Put it on a canvas
    canvas = snx.Canvas(width=int(view.layout.width), height=int(view.layout.height))
    canvas.views.append(view)

    # Mouse over that image in the top right corner
    canvas_pos = (view.layout.width, 0)
    world_ray = canvas.to_world(canvas_pos)
    assert world_ray is not None
    event = MouseMoveEvent(
        canvas_pos=canvas_pos, world_ray=world_ray, buttons=MouseButton.NONE
    )

    # And show both the view and the image saw the event
    canvas.handle(event)
    view_filter.assert_called_once_with(event)
    img_filter.assert_called_once_with(event, img)

    # Reset the mocks
    img_filter.reset_mock()
    view_filter.reset_mock()

    # Mouse over empty space in the top left corner
    canvas_pos = (0, 0)
    world_ray = canvas.to_world(canvas_pos)
    assert world_ray is not None
    event = MouseMoveEvent(
        canvas_pos=canvas_pos, world_ray=world_ray, buttons=MouseButton.NONE
    )

    # And show that the image did not see the event
    # but that the view still saw the event
    canvas.handle(event)
    img_filter.assert_not_called()
    view_filter.assert_called_once_with(event)
