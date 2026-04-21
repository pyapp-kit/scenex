"""Tests pertaining to Jupyter event generation"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from app_model.types import KeyBinding

import scenex as snx
from scenex.adaptors._auto import determine_backend
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
from scenex.model._transform import Transform

if TYPE_CHECKING:
    from scenex.adaptors._base import CanvasAdaptor

if determine_app() != GuiFrontend.JUPYTER:
    pytest.skip(
        "Skipping Jupyter tests as Jupyter will not be used in this environment",
        allow_module_level=True,
    )

# HACK: Enable tests inside vispy
if determine_backend() == "vispy":
    import asyncio
    import os

    os.environ["_VISPY_TESTING_APP"] = "jupyter_rfb"
    asyncio.set_event_loop(asyncio.new_event_loop())

    os.environ["SCENEX_APP_BACKEND"] = "jupyter"


@pytest.fixture
def evented_canvas() -> snx.Canvas:
    # IPython.getIPython().run_line_magic("gui", "inline")
    camera = snx.Camera(transform=Transform(), interactive=True)
    scene = snx.Scene(children=[])
    view = snx.View(scene=scene, camera=camera)
    canvas = snx.Canvas(views=[view])
    return canvas


# See jupyter_rfb.events
NONE = 0
LEFT_MOUSE = 1
RIGHT_MOUSE = 2


def test_pointer_down(evented_canvas: snx.Canvas) -> None:
    snx.show(evented_canvas)
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    press_point = (0, 0)
    # Press the left button
    native.handle_event(
        {
            "event_type": "pointer_down",
            "x": press_point[0],
            "y": press_point[1],
            "button": LEFT_MOUSE,
        }
    )
    mock_filter.assert_called_once_with(
        MousePressEvent(pos=press_point, buttons=MouseButton.LEFT)
    )

    mock_filter.reset_mock()
    # Now press the right button as well
    native.handle_event(
        {
            "event_type": "pointer_down",
            "x": press_point[0],
            "y": press_point[1],
            "button": RIGHT_MOUSE,
        }
    )
    mock_filter.assert_called_once_with(
        MousePressEvent(pos=press_point, buttons=MouseButton.RIGHT)
    )


def test_pointer_up(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    press_point = (0, 0)
    native.handle_event(
        {
            "event_type": "pointer_up",
            "x": press_point[0],
            "y": press_point[1],
            "button": LEFT_MOUSE,
        }
    )
    mock_filter.assert_called_once_with(
        MouseReleaseEvent(pos=press_point, buttons=MouseButton.LEFT)
    )


def test_pointer_move(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    press_point = (0, 0)
    native.handle_event(
        {
            "event_type": "pointer_move",
            "x": press_point[0],
            "y": press_point[1],
            "button": LEFT_MOUSE,
        }
    )
    mock_filter.assert_called_once_with(
        MouseMoveEvent(pos=press_point, buttons=MouseButton.LEFT)
    )
    mock_filter.reset_mock()

    native.handle_event(
        {
            "event_type": "pointer_move",
            "x": press_point[0],
            "y": press_point[1],
            "buttons": (LEFT_MOUSE, RIGHT_MOUSE),
        }
    )
    mock_filter.assert_called_once_with(
        MouseMoveEvent(pos=press_point, buttons=MouseButton.LEFT | MouseButton.RIGHT)
    )


def test_mouse_double_click(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    press_point = (0, 0)
    native.handle_event(
        {
            "event_type": "double_click",
            "x": press_point[0],
            "y": press_point[1],
            "button": LEFT_MOUSE,
        }
    )
    mock_filter.assert_called_once_with(
        MouseDoublePressEvent(pos=press_point, buttons=MouseButton.LEFT)
    )


def test_wheel(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    press_point = (0, 0)
    native.handle_event(
        {
            "event_type": "wheel",
            "x": press_point[0],
            "y": press_point[1],
            "dx": 0,
            "dy": -120,  # Note that Jupyter_rfb uses a different y convention
        }
    )
    mock_filter.assert_called_once_with(
        WheelEvent(pos=press_point, buttons=MouseButton.NONE, angle_delta=(0, 120))
    )


def test_resize(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()

    new_size = (400, 300)
    assert evented_canvas.width != new_size[0]
    assert evented_canvas.height != new_size[1]
    native.handle_event(
        {
            "event_type": "resize",
            "width": new_size[0],
            "height": new_size[1],
            "pixel_ratio": 1.0,
        }
    )
    assert evented_canvas.width == new_size[0]
    assert evented_canvas.height == new_size[1]


def test_pointer_enter(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    enter_point = (0, 0)
    native.handle_event(
        {
            "event_type": "pointer_enter",
            "x": enter_point[0],
            "y": enter_point[1],
            "button": NONE,
        }
    )

    # Verify MouseEnterEvent was passed to Canvas.handle
    mock_filter.assert_called_once_with(
        MouseEnterEvent(pos=enter_point, buttons=MouseButton.NONE)
    )


def test_pointer_leave(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    enter_point = (0, 0)
    native.handle_event(
        {
            "event_type": "pointer_enter",
            "x": enter_point[0],
            "y": enter_point[1],
            "button": NONE,
        }
    )
    mock_filter.reset_mock()

    # Now leave
    native.handle_event({"event_type": "pointer_leave"})

    # Verify MouseLeaveEvent was passed to Canvas.handle
    mock_filter.assert_called_once_with(MouseLeaveEvent())


def test_set_cursor(evented_canvas: snx.Canvas) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    snx.set_cursor(evented_canvas, CursorType.CROSS)
    assert native.cursor == "crosshair"


def test_key_event(evented_canvas: snx.Canvas) -> None:
    snx.show(evented_canvas)
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    native.handle_event({"event_type": "key_down", "key": "a", "modifiers": []})
    native.handle_event({"event_type": "key_up", "key": "a", "modifiers": []})

    assert mock_filter.call_args_list[0].args == (
        KeyPressEvent(key=KeyBinding.from_str("A")),
    )
    assert mock_filter.call_args_list[1].args == (
        KeyReleaseEvent(key=KeyBinding.from_str("A")),
    )
