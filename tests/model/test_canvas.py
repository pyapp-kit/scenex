from unittest.mock import Mock, call

import scenex as snx
from scenex.app.events import (
    MouseButton,
    MouseEnterEvent,
    MouseLeaveEvent,
    MouseMoveEvent,
)


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


def test_handle_enter_events() -> None:
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
