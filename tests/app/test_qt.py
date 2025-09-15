"""Tests pertaining to VisPy adaptors"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

import scenex as snx
from scenex.app import GuiFrontend, app, determine_app
from scenex.app.events import (
    MouseButton,
    MouseDoublePressEvent,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
    Ray,
)
from scenex.model._transform import Transform

if TYPE_CHECKING:
    from scenex.adaptors._base import CanvasAdaptor

if determine_app() == GuiFrontend.QT:
    from qtpy.QtCore import QPoint, Qt

    if TYPE_CHECKING:
        from pytestqt.qtbot import QtBot  # pyright: ignore[reportMissingImports]
        from qtpy.QtWidgets import QWidget
else:
    pytest.skip(
        "Skipping Qt tests as Qt will not be used in this environment",
        allow_module_level=True,
    )


@pytest.fixture
def evented_canvas(qtbot: QtBot) -> snx.Canvas:
    camera = snx.Camera(transform=Transform(), interactive=True)
    scene = snx.Scene(children=[])
    view = snx.View(scene=scene, camera=camera)
    canvas = snx.Canvas()
    canvas.views.append(view)
    native = cast(
        "CanvasAdaptor", canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    qtbot.addWidget(native)
    return canvas


def _validate_ray(maybe_ray: Ray | None) -> Ray:
    assert maybe_ray is not None
    return maybe_ray


def test_mouse_press(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    # Press the left button
    qtbot.mousePress(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
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
    qtbot.mousePress(native, Qt.MouseButton.RightButton, pos=QPoint(*press_point))
    mock.assert_called_once_with(
        MousePressEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.RIGHT,
        ),
        evented_canvas.views[0].camera,
    )


def test_mouse_release(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    qtbot.mouseRelease(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
    mock.assert_called_once_with(
        MouseReleaseEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT,
        ),
        evented_canvas.views[0].camera,
    )


def test_mouse_move(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    # FIXME: For some reason the mouse press is necessary for processing events?
    qtbot.mousePress(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
    qtbot.mousePress(native, Qt.MouseButton.RightButton, pos=QPoint(*press_point))
    mock.reset_mock()
    qtbot.mouseMove(native, pos=QPoint(*press_point))
    mock.assert_called_once_with(
        MouseMoveEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT | MouseButton.RIGHT,
        ),
        evented_canvas.views[0].camera,
    )


def test_mouse_click(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    qtbot.mouseClick(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
    assert mock.call_args_list[0].args == (
        MousePressEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT,
        ),
        evented_canvas.views[0].camera,
    )
    assert mock.call_args_list[1].args == (
        MouseReleaseEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT,
        ),
        evented_canvas.views[0].camera,
    )


def test_mouse_double_click(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    press_point = (5, 10)
    # Note that in Qt a double click does NOT implicitly imply a release as well.
    qtbot.mouseDClick(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
    assert mock.call_args_list[0].args == (
        MouseDoublePressEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT,
        ),
        evented_canvas.views[0].camera,
    )


def test_resize(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    qtbot.add_widget(native)
    mock = MagicMock()
    evented_canvas.views[0].camera.set_event_filter(mock)
    new_size = (400, 300)
    assert evented_canvas.width != new_size[0]
    assert evented_canvas.height != new_size[1]
    # Note that the widget must be visible for a resize event to fire
    cast("QWidget", native).setVisible(True)
    cast("QWidget", native).resize(*new_size)
    app().process_events()
    assert evented_canvas.width == new_size[0]
    assert evented_canvas.height == new_size[1]


# TODO: Implement when Qt new enough
# https://doc.qt.io/qt-6/qtest.html#wheelEvent
# def test_wheel(evented_canvas: snx.Canvas):
#     pass
