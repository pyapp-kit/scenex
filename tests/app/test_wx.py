"""Tests pertaining to WxPython canvas events."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from app_model.types import KeyBinding

import scenex as snx
from scenex.app import CursorType, GuiFrontend, determine_app
from scenex.app.events import (
    KeyPressEvent,
    KeyReleaseEvent,
    MouseButton,
    MouseDoublePressEvent,
    MouseEnterEvent,
    MouseLeaveEvent,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
    WheelEvent,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from scenex.adaptors._base import CanvasAdaptor

if determine_app() == GuiFrontend.WX:
    import wx
else:
    pytest.skip(
        "Skipping WxPython tests as WxPython will not be used in this environment",
        allow_module_level=True,
    )


@pytest.fixture
def evented_canvas(basic_view: snx.Scene) -> Iterator[snx.Canvas]:
    canvas = snx.show(basic_view)
    yield canvas
    # FIXME: Probably good to destroy the canvas here - we may need a method for that


def _processEvent(
    evt: wx.PyEventBinder,
    wdg: wx.Control,
    *,
    left_down: bool | None = None,
    right_down: bool | None = None,
    middle_down: bool | None = None,
    **kwargs: Any,
) -> None:
    """Simulates a wx event.

    Note that wx.UIActionSimulator is an alternative to this approach.
    It seems to actually move the cursor around though, which is really annoying :)

    For mouse events, pass left_down/right_down/middle_down to explicitly set which
    buttons are held. When omitted, each flag is inferred from the event type alone
    (True only for the corresponding DOWN event), so callers must pass explicit values
    when multiple buttons are involved.
    """
    if evt == wx.EVT_SIZE:
        ev = wx.SizeEvent(kwargs["sz"], evt.typeId)
    elif evt in (wx.EVT_KEY_DOWN, wx.EVT_KEY_UP):
        ev = wx.KeyEvent(evt.typeId)
        ev.SetKeyCode(kwargs["keyCode"])
    else:
        ev = wx.MouseEvent(evt.typeId)
        ev.SetPosition(kwargs["pos"])
        if rot := kwargs.get("rot"):
            ev.SetWheelRotation(rot[1])
        ev.leftIsDown = (
            left_down
            if left_down is not None
            else evt in (wx.EVT_LEFT_DOWN, wx.EVT_LEFT_DCLICK)
        )
        ev.rightIsDown = (
            right_down
            if right_down is not None
            else evt in (wx.EVT_RIGHT_DOWN, wx.EVT_RIGHT_DCLICK)
        )
        ev.middleIsDown = (
            middle_down
            if middle_down is not None
            else evt in (wx.EVT_MIDDLE_DOWN, wx.EVT_MIDDLE_DCLICK)
        )

    wx.PostEvent(wdg.GetEventHandler(), ev)
    # Borrowed from:
    # https://github.com/wxWidgets/Phoenix/blob/master/unittests/wtc.py#L41
    wdg.Show(True)
    wx.MilliSleep(50)
    evtLoop = wx.App.Get().GetTraits().CreateEventLoop()
    wx.EventLoopActivator(evtLoop)
    evtLoop.YieldFor(wx.EVT_CATEGORY_ALL)  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(
    ("evt", "button"),
    [
        (wx.EVT_LEFT_DCLICK, MouseButton.LEFT),
        (wx.EVT_RIGHT_DCLICK, MouseButton.RIGHT),
        (wx.EVT_MIDDLE_DCLICK, MouseButton.MIDDLE),
    ],
)
def test_mouse_double_click(
    evented_canvas: snx.Canvas, evt: wx.PyEventBinder, button: MouseButton
) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock(return_value=False)
    evented_canvas.set_event_filter(mock_filter)

    press_point = (5, 10)
    _processEvent(evt, native, pos=wx.Point(*press_point))
    mock_filter.assert_called_once_with(
        MouseDoublePressEvent(pos=press_point, buttons=button)
    )


def test_mouse_double_click_after_press(evented_canvas: snx.Canvas) -> None:
    """Double clicks use <button>IsDown to determine active buttons, which means that we
    need to be careful when handling these events to ensure that the correct button is
    reported when multiple are pressed."""
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock(return_value=False)
    evented_canvas.set_event_filter(mock_filter)

    press_point = (5, 10)
    # Press the left button
    _processEvent(wx.EVT_LEFT_DOWN, native, pos=wx.Point(*press_point))
    mock_filter.reset_mock()
    # Now double click - we should still only have the left button active
    _processEvent(
        wx.EVT_RIGHT_DCLICK, native, pos=wx.Point(*press_point), left_down=True
    )
    mock_filter.assert_called_once_with(
        MouseDoublePressEvent(pos=press_point, buttons=MouseButton.RIGHT)
    )


@pytest.mark.parametrize(
    ("evt", "button"),
    [
        (wx.EVT_LEFT_DOWN, MouseButton.LEFT),
        (wx.EVT_RIGHT_DOWN, MouseButton.RIGHT),
        (wx.EVT_MIDDLE_DOWN, MouseButton.MIDDLE),
    ],
)
def test_mouse_press(
    evented_canvas: snx.Canvas, evt: wx.PyEventBinder, button: MouseButton
) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock(return_value=False)
    evented_canvas.set_event_filter(mock_filter)

    press_point = (5, 10)
    _processEvent(evt, native, pos=wx.Point(*press_point))
    mock_filter.assert_called_once_with(
        MousePressEvent(pos=press_point, buttons=button)
    )


def test_mouse_release(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock(return_value=False)
    evented_canvas.set_event_filter(mock_filter)

    press_point = (5, 10)
    _processEvent(wx.EVT_LEFT_UP, native, pos=wx.Point(*press_point))
    mock_filter.assert_called_once_with(
        MouseReleaseEvent(pos=press_point, buttons=MouseButton.LEFT)
    )


def test_multiple_mouse_release(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock(return_value=False)
    evented_canvas.set_event_filter(mock_filter)

    press_point = (5, 10)
    # Press the left button
    _processEvent(wx.EVT_LEFT_DOWN, native, pos=wx.Point(*press_point))
    # Press the right button
    _processEvent(wx.EVT_RIGHT_DOWN, native, pos=wx.Point(*press_point), left_down=True)
    # Now release just the left button; right remains held
    _processEvent(wx.EVT_LEFT_UP, native, pos=wx.Point(*press_point), right_down=True)

    mock_filter.reset_mock()
    # Now move the mouse - we should only have the right button active
    move_point = (6, 11)
    _processEvent(wx.EVT_MOTION, native, pos=wx.Point(*move_point), right_down=True)
    mock_filter.assert_called_once_with(
        MouseMoveEvent(pos=move_point, buttons=MouseButton.RIGHT)
    )


def test_mouse_move(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock(return_value=False)
    evented_canvas.set_event_filter(mock_filter)

    press_point = (5, 10)
    # FIXME: For some reason the mouse press is necessary for processing events?
    _processEvent(wx.EVT_LEFT_DOWN, native, pos=wx.Point(*press_point))
    _processEvent(wx.EVT_RIGHT_DOWN, native, pos=wx.Point(*press_point), left_down=True)
    mock_filter.reset_mock()
    _processEvent(
        wx.EVT_MOTION,
        native,
        pos=wx.Point(*press_point),
        left_down=True,
        right_down=True,
    )
    mock_filter.assert_called_once_with(
        MouseMoveEvent(pos=press_point, buttons=MouseButton.LEFT | MouseButton.RIGHT)
    )


def test_mouse_wheel(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock(return_value=False)
    evented_canvas.set_event_filter(mock_filter)

    press_point = (5, 10)
    _processEvent(wx.EVT_MOUSEWHEEL, native, pos=wx.Point(*press_point), rot=(0, 120))
    mock_filter.assert_called_once_with(
        WheelEvent(pos=press_point, buttons=MouseButton.NONE, angle_delta=(0, 120))
    )


def test_resize(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()

    new_size = (400, 300)
    # Note that the widget must be visible for a resize event to fire
    _processEvent(wx.EVT_SIZE, native, sz=wx.Size(*new_size))
    assert evented_canvas.width == new_size[0]
    assert evented_canvas.height == new_size[1]


def test_mouse_enter(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock(return_value=False)
    evented_canvas.set_event_filter(mock_filter)

    enter_point = (0, 15)
    _processEvent(wx.EVT_ENTER_WINDOW, native, pos=wx.Point(*enter_point))

    # Verify MouseEnterEvent was passed to Canvas.handle
    mock_filter.assert_called_once_with(
        MouseEnterEvent(pos=enter_point, buttons=MouseButton.NONE)
    )


def test_mouse_leave(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock(return_value=False)
    evented_canvas.set_event_filter(mock_filter)

    # NOTE: We need to first enter to establish the view as active
    enter_point = (0, 15)
    _processEvent(wx.EVT_ENTER_WINDOW, native, pos=wx.Point(*enter_point))
    mock_filter.reset_mock()

    # Now leave
    _processEvent(wx.EVT_LEAVE_WINDOW, native, pos=wx.Point(0, 0))

    # Verify MouseLeaveEvent was passed to Canvas.handle
    mock_filter.assert_called_once_with(MouseLeaveEvent())


def test_set_cursor(evented_canvas: snx.Canvas) -> None:
    adaptor = cast("CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0])
    native = cast("wx.Window", adaptor._snx_get_native())
    # Wx doesn't really give us a way to assert the right thing happened...
    # ...the best we can do is assert a change.
    old = native.GetCursor()
    snx.set_cursor(evented_canvas, CursorType.CROSS)
    assert not native.GetCursor().IsSameAs(old)


def test_key_event(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock(return_value=False)
    evented_canvas.set_event_filter(mock_filter)

    _processEvent(wx.EVT_KEY_DOWN, native, keyCode=ord("A"))
    _processEvent(wx.EVT_KEY_UP, native, keyCode=ord("A"))

    assert mock_filter.call_args_list[0].args == (
        KeyPressEvent(key=KeyBinding.from_str("A")),
    )
    assert mock_filter.call_args_list[1].args == (
        KeyReleaseEvent(key=KeyBinding.from_str("A")),
    )
