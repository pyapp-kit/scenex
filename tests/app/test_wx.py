"""Tests pertaining to WxPython canvas events."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

import scenex as snx
from scenex.app import GuiFrontend, determine_app
from scenex.app.events import (
    MouseButton,
    MouseEnterEvent,
    MouseLeaveEvent,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
    Ray,
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


def _processEvent(evt: wx.PyEventBinder, wdg: wx.Control, **kwargs: Any) -> None:
    """Simulates a wx event.

    Note that wx.UIActionSimulator is an alternative to this approach.
    It seems to actually move the cursor around though, which is really annoying :)
    """
    if evt == wx.EVT_SIZE:
        ev = wx.SizeEvent(kwargs["sz"], evt.typeId)
    else:
        ev = wx.MouseEvent(evt.typeId)
        ev.SetPosition(kwargs["pos"])
        if rot := kwargs.get("rot"):
            ev.SetWheelRotation(rot[1])
        ev.SetLeftDown(True)

    wx.PostEvent(wdg.GetEventHandler(), ev)
    # Borrowed from:
    # https://github.com/wxWidgets/Phoenix/blob/master/unittests/wtc.py#L41
    wdg.Show(True)
    wx.MilliSleep(50)
    evtLoop = wx.App.Get().GetTraits().CreateEventLoop()
    wx.EventLoopActivator(evtLoop)
    evtLoop.YieldFor(wx.EVT_CATEGORY_ALL)  # pyright: ignore[reportAttributeAccessIssue]


def _validate_ray(maybe_ray: Ray | None) -> Ray:
    assert maybe_ray is not None
    return maybe_ray


def test_mouse_press(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    # Press the left button
    _processEvent(wx.EVT_LEFT_DOWN, native, pos=wx.Point(*press_point))
    mock.assert_called_once_with(
        MousePressEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT,
        ),
        evented_canvas.views[0].camera,
    )
    mock.reset_mock()

    # Now press the right button
    _processEvent(wx.EVT_RIGHT_DOWN, native, pos=wx.Point(*press_point))
    mock.assert_called_once_with(
        MousePressEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.RIGHT,
        ),
        evented_canvas.views[0].camera,
    )


def test_mouse_release(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    _processEvent(wx.EVT_LEFT_UP, native, pos=wx.Point(*press_point))
    mock.assert_called_once_with(
        MouseReleaseEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT,
        ),
        evented_canvas.views[0].camera,
    )


def test_mouse_move(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    # FIXME: For some reason the mouse press is necessary for processing events?
    _processEvent(wx.EVT_LEFT_DOWN, native, pos=wx.Point(*press_point))
    _processEvent(wx.EVT_RIGHT_DOWN, native, pos=wx.Point(*press_point))
    mock.reset_mock()
    _processEvent(wx.EVT_MOTION, native, pos=wx.Point(*press_point))
    mock.assert_called_once_with(
        MouseMoveEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT | MouseButton.RIGHT,
        ),
        evented_canvas.views[0].camera,
    )


def test_mouse_wheel(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    _processEvent(wx.EVT_MOUSEWHEEL, native, pos=wx.Point(*press_point), rot=(0, 120))
    mock.assert_called_once_with(
        WheelEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.NONE,
            angle_delta=(0, 120),
        ),
        evented_canvas.views[0].camera,
    )


def test_resize(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    new_size = (400, 300)
    # Note that the widget must be visible for a resize event to fire
    _processEvent(wx.EVT_SIZE, native, sz=wx.Size(*new_size))
    assert evented_canvas.width == new_size[0]
    assert evented_canvas.height == new_size[1]


def test_mouse_enter(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    view_mock = MagicMock()
    evented_canvas.views[0].set_event_filter(view_mock)
    enter_point = (0, 15)
    _processEvent(wx.EVT_ENTER_WINDOW, native, pos=wx.Point(*enter_point))

    # Verify MouseEnterEvent was passed to view filter
    view_mock.assert_called_once_with(
        MouseEnterEvent(
            canvas_pos=enter_point,
            world_ray=_validate_ray(evented_canvas.to_world(enter_point)),
            buttons=MouseButton.NONE,
        )
    )


def test_mouse_leave(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    view_mock = MagicMock()
    evented_canvas.views[0].set_event_filter(view_mock)
    # NOTE: We need to first enter to establish the view as active
    enter_point = (0, 15)
    _processEvent(wx.EVT_ENTER_WINDOW, native, pos=wx.Point(*enter_point))
    view_mock.reset_mock()

    # Now leave
    _processEvent(wx.EVT_LEAVE_WINDOW, native, pos=wx.Point(0, 0))
    # Verify MouseLeaveEvent was passed to view filter
    view_mock.assert_called_once_with(MouseLeaveEvent())
