from unittest.mock import Mock, call

import cmap
import numpy as np

import scenex as snx
from scenex.app.events import (
    MouseButton,
    MouseEnterEvent,
    MouseLeaveEvent,
    MouseMoveEvent,
)
from scenex.utils.projections import orthographic


def test_multiple_views() -> None:
    # Create a canvas with two views
    view1 = snx.View()  # Left half
    view1.layout.x = "0%", "50%"
    view2 = snx.View()  # Right half
    view2.layout.x = "50%", "100%"
    canvas = snx.Canvas(views=[view1, view2])

    x1, y1, w1, h1 = canvas.rect_for(view1)
    x2, y2, w2, h2 = canvas.rect_for(view2)

    # By default the views are equally sized and side-by-side
    assert w1 == w2
    assert h1 == h2
    assert x1 + w1 == x2
    assert y1 == y2

    # Changing the canvas size should preserve the equal-split relationship
    canvas.width = canvas.width // 2
    canvas.height = canvas.height * 2

    x1, y1, w1, h1 = canvas.rect_for(view1)
    x2, y2, w2, h2 = canvas.rect_for(view2)
    assert w1 == w2
    assert h1 == h2
    assert x1 + w1 == x2
    assert y1 == y2


def test_event_filter() -> None:
    """Tests the ability to set a canvas-level event filter."""
    view = snx.View()
    view_filter = Mock()
    view.set_event_filter(view_filter)

    canvas = snx.Canvas(views=[view])
    canvas_filter = Mock()
    canvas_filter.return_value = False
    canvas.set_event_filter(canvas_filter)
    # Ensure that the canvas can receive events
    evt = MouseMoveEvent(pos=(0, 0), buttons=MouseButton.NONE)
    canvas.handle(evt)
    canvas_filter.assert_called_with(evt)
    view_filter.assert_called_with(evt)

    # Ensure that the canvas can block events if the filter returns True
    view_filter.reset_mock()
    canvas_filter.reset_mock()
    canvas_filter.return_value = True

    canvas.handle(evt)
    canvas_filter.assert_called_with(evt)
    view_filter.assert_not_called()


def test_handle_view_events() -> None:
    """Tests inter-view mouse event handling.

    Note that this is different from testing that events are passed to the handler,
    which is done in the app package.
    """
    # Create a canvas with two views
    view1 = snx.View()  # Left half
    view1.layout.x = "0%", "50%"
    view2 = snx.View()  # Right half
    view2.layout.x = "50%", "100%"
    canvas = snx.Canvas(views=[view1, view2])
    mock_filter = Mock()
    view1.set_event_filter(mock_filter)

    # Assert MouseEnterEvents are directed to the correct view
    evt = MouseEnterEvent(pos=(0, 0), buttons=MouseButton.NONE)
    canvas.handle(evt)
    mock_filter.assert_called_once_with(evt)
    mock_filter.reset_mock()

    # Assert MouseLeaveEvents are directed to the correct view
    evt = MouseLeaveEvent()
    canvas.handle(evt)
    mock_filter.assert_called_once_with(evt)
    mock_filter.reset_mock()

    # Assert MouseEnterEvents are generated if another event type is sent to a new view
    evt = MouseMoveEvent(pos=(2, 0), buttons=MouseButton.NONE)
    canvas.handle(evt)
    assert mock_filter.call_count == 2
    assert mock_filter.call_args_list[0] == call(
        MouseEnterEvent(pos=evt.pos, buttons=MouseButton.NONE)
    )
    assert mock_filter.call_args_list[1] == call(evt)
    mock_filter.reset_mock()

    # Assert MouseEnterEvents are generated when moving between views
    mock_filter2 = Mock()
    view2.set_event_filter(mock_filter2)
    evt = MouseMoveEvent(pos=(canvas.width - 1, 0), buttons=MouseButton.NONE)
    canvas.handle(evt)
    mock_filter.assert_called_once_with(MouseLeaveEvent())
    assert mock_filter2.call_count == 2
    assert mock_filter2.call_args_list[0] == call(
        MouseEnterEvent(pos=evt.pos, buttons=MouseButton.NONE)
    )
    assert mock_filter2.call_args_list[1] == call(evt)


def test_render() -> None:
    """Smoke test for canvas.render()."""

    # Scene: a red quad at world (±0.5, ±0.5, 0), viewed through an orthographic
    # camera mapping world ±1 → NDC ±1 via orthographic(2, 2, 2).  The mesh
    # therefore occupies NDC ±0.5, which is the centre 50 % of the viewport.
    vertices = np.array(
        [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = snx.Mesh(
        vertices=vertices,
        faces=faces,
        color=snx.UniformColor(color=cmap.Color("red")),
    )

    camera = snx.Camera(
        transform=snx.Transform(),
        projection=orthographic(2, 2, 2),
    )
    scene = snx.Scene(children=[mesh])
    view = snx.View(scene=scene, camera=camera)
    canvas = snx.Canvas(
        views=[view], width=400, height=400, background_color=cmap.Color("black")
    )

    # Render the image
    img = canvas.render()
    assert isinstance(img, np.ndarray)
    assert img.shape == (canvas.height, canvas.width, 4)

    # Test the center of the rendered view, which should be very red.
    center = (canvas.height // 2, canvas.width // 2)
    center_rgb = img[center[0], center[1], :3]
    np.testing.assert_array_equal(center_rgb, [255, 0, 0])

    # Test the corners of the rendered view, which should be the background color.
    background_rgb = np.array(cmap.Color("black").rgba[:3])
    for pixel in [
        img[0, 0],
        img[canvas.height - 1, canvas.width - 1],
    ]:
        corner_rgb = pixel[:3]
        np.testing.assert_array_equal(corner_rgb, background_rgb)

    canvas.close()
