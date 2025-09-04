"""Tests pertaining to Jupyter event generation"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

import scenex as snx
from scenex.adaptors._auto import determine_backend
from scenex.app import GuiFrontend, determine_app
from scenex.app.events import MouseButton, MouseEvent, Ray, WheelEvent
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

    os.environ["SCENEX_WIDGET_BACKEND"] = "jupyter"


@pytest.fixture
def evented_canvas() -> snx.Canvas:
    # IPython.getIPython().run_line_magic("gui", "inline")
    camera = snx.Camera(transform=Transform(), interactive=True)
    scene = snx.Scene(children=[])
    view = snx.View(scene=scene, camera=camera)
    canvas = snx.Canvas()
    canvas.views.append(view)
    return canvas


def _validate_ray(maybe_ray: Ray | None) -> Ray:
    assert maybe_ray is not None
    return maybe_ray


# See jupyter_rfb.events
LEFT_MOUSE = 1
RIGHT_MOUSE = 2


def test_pointer_down(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
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

    # Now press the right button as well
    native.handle_event(
        {
            "event_type": "pointer_down",
            "x": press_point[0],
            "y": press_point[1],
            "button": RIGHT_MOUSE,
        }
    )
    mock.assert_called_once_with(
        MouseEvent(
            "press",
            press_point,
            _validate_ray(evented_canvas.to_world(press_point)),
            MouseButton.RIGHT,
        ),
        evented_canvas.views[0].camera,
    )


def test_pointer_up(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (0, 0)
    native.handle_event(
        {
            "event_type": "pointer_up",
            "x": press_point[0],
            "y": press_point[1],
            "button": LEFT_MOUSE,
        }
    )
    mock.assert_called_once_with(
        MouseEvent(
            "release",
            press_point,
            _validate_ray(evented_canvas.to_world(press_point)),
            MouseButton.LEFT,
        ),
        evented_canvas.views[0].camera,
    )


def test_pointer_move(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (0, 0)
    native.handle_event(
        {
            "event_type": "pointer_move",
            "x": press_point[0],
            "y": press_point[1],
            "button": LEFT_MOUSE,
        }
    )
    mock.assert_called_once_with(
        MouseEvent(
            "move",
            press_point,
            _validate_ray(evented_canvas.to_world(press_point)),
            MouseButton.LEFT,
        ),
        evented_canvas.views[0].camera,
    )
    mock.reset_mock()

    native.handle_event(
        {
            "event_type": "pointer_move",
            "x": press_point[0],
            "y": press_point[1],
            "buttons": (LEFT_MOUSE, RIGHT_MOUSE),
        }
    )
    mock.assert_called_once_with(
        MouseEvent(
            "move",
            press_point,
            _validate_ray(evented_canvas.to_world(press_point)),
            MouseButton.LEFT | MouseButton.RIGHT,
        ),
        evented_canvas.views[0].camera,
    )


def test_mouse_double_click(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (0, 0)
    native.handle_event(
        {
            "event_type": "double_click",
            "x": press_point[0],
            "y": press_point[1],
            "button": LEFT_MOUSE,
        }
    )
    mock.assert_called_once_with(
        MouseEvent(
            "double_press",
            press_point,
            _validate_ray(evented_canvas.to_world(press_point)),
            MouseButton.LEFT,
        ),
        evented_canvas.views[0].camera,
    )


def test_wheel(evented_canvas: snx.Canvas) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
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
