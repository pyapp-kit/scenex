"""Tests pertaining to WxPython canvas events."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

import scenex as snx
from scenex.events._auto import GuiFrontend, app, determine_app
from scenex.events.events import MouseButton, MouseEvent, Ray, WheelEvent
from scenex.model._transform import Transform

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


@pytest.fixture(scope="session")
def wxapp() -> Iterator[wx.App]:
    yield app().create_app()


@pytest.fixture
def evented_canvas() -> snx.Canvas:
    camera = snx.Camera(transform=Transform(), interactive=True)
    scene = snx.Scene(children=[])
    canvas = snx.Canvas()
    view = snx.View(scene=scene, camera=camera)
    # FIXME: The canvas should take care of setting this.
    view.canvas = canvas
    canvas.views.append(view)
    canvas._get_adaptors(create=True)[0]._snx_get_native()
    return canvas


def _processEvent(
    wxapp: wx.App, evt: wx.PyEventBinder, wdg: wx.Control, **kwargs: Any
) -> None:
    """Simulates a wx event.

    Note that wx.UIActionSimulator is an alternative to this approach.
    It seems to actually move the cursor around though, which is really annoying :)
    """
    if evt == wx.EVT_ACTIVATE:
        active = kwargs.get("active", True)
        ev = wx.ActivateEvent(eventType=evt.typeId, active=active)
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
    evtLoop = wxapp.GetTraits().CreateEventLoop()
    wx.EventLoopActivator(evtLoop)
    evtLoop.YieldFor(wx.EVT_CATEGORY_ALL)  # pyright: ignore[reportAttributeAccessIssue]


def _validate_ray(maybe_ray: Ray | None) -> Ray:
    assert maybe_ray is not None
    return maybe_ray


def test_mouse_press(wxapp: wx.App, evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_window_ref()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    # Press the left button
    _processEvent(wxapp, wx.EVT_LEFT_DOWN, native, pos=wx.Point(*press_point))
    mock.assert_called_once_with(
        MouseEvent(
            "press",
            press_point,
            _validate_ray(evented_canvas.to_world(press_point)),
            MouseButton.LEFT,
        ),
        evented_canvas.views[0].camera,
    )
    mock.reset_mock()

    # Now press the right button
    _processEvent(wxapp, wx.EVT_RIGHT_DOWN, native, pos=wx.Point(*press_point))
    mock.assert_called_once_with(
        MouseEvent(
            "press",
            press_point,
            _validate_ray(evented_canvas.to_world(press_point)),
            MouseButton.RIGHT,
        ),
        evented_canvas.views[0].camera,
    )


def test_mouse_release(wxapp: wx.App, evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_window_ref()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    _processEvent(wxapp, wx.EVT_LEFT_UP, native, pos=wx.Point(*press_point))
    mock.assert_called_once_with(
        MouseEvent(
            "release",
            press_point,
            _validate_ray(evented_canvas.to_world(press_point)),
            MouseButton.LEFT,
        ),
        evented_canvas.views[0].camera,
    )


def test_mouse_move(wxapp: wx.App, evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_window_ref()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    # FIXME: For some reason the mouse press is necessary for processing events?
    _processEvent(wxapp, wx.EVT_LEFT_DOWN, native, pos=wx.Point(*press_point))
    _processEvent(wxapp, wx.EVT_RIGHT_DOWN, native, pos=wx.Point(*press_point))
    mock.reset_mock()
    _processEvent(wxapp, wx.EVT_MOTION, native, pos=wx.Point(*press_point))
    mock.assert_called_once_with(
        MouseEvent(
            "move",
            press_point,
            _validate_ray(evented_canvas.to_world(press_point)),
            MouseButton.LEFT | MouseButton.RIGHT,
        ),
        evented_canvas.views[0].camera,
    )


def test_mouse_wheel(wxapp: wx.App, evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_window_ref()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    _processEvent(
        wxapp, wx.EVT_MOUSEWHEEL, native, pos=wx.Point(*press_point), rot=(0, 120)
    )
    mock.assert_called_once_with(
        WheelEvent(
            "wheel",
            press_point,
            _validate_ray(evented_canvas.to_world(press_point)),
            MouseButton.NONE,
            angle_delta=(0, 120),
        ),
        evented_canvas.views[0].camera,
    )
