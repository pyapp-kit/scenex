"""Tests pertaining to VisPy adaptors"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

import scenex as snx
from scenex.app import GuiFrontend, app, determine_app
from scenex.app.events import (
    MouseButton,
    MouseDoublePressEvent,
    MouseEnterEvent,
    MouseLeaveEvent,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
    Ray,
)
from scenex.model._transform import Transform

if TYPE_CHECKING:
    from scenex.adaptors._base import CanvasAdaptor

if determine_app() == GuiFrontend.QT:
    from qtpy.QtCore import QEvent, QPoint, QPointF, Qt
    from qtpy.QtGui import QEnterEvent
    from qtpy.QtWidgets import QApplication

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
    """Test that Qt mouse press events are converted and given to the canvas."""
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    press_point = (5, 10)
    with patch("scenex.model._canvas.Canvas.handle") as mock:
        # Press the left button
        qtbot.mousePress(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
    mock.assert_called_once_with(
        MousePressEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT,
        )
    )

    # Now press the right button
    with patch("scenex.model._canvas.Canvas.handle") as mock:
        qtbot.mousePress(native, Qt.MouseButton.RightButton, pos=QPoint(*press_point))
    mock.assert_called_once_with(
        MousePressEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.RIGHT,
        )
    )


def test_mouse_release(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    """Test that Qt mouse release events are converted and given to the canvas."""
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    press_point = (5, 10)
    with patch("scenex.model._canvas.Canvas.handle") as mock:
        qtbot.mouseRelease(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
    mock.assert_called_once_with(
        MouseReleaseEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT,
        )
    )


def test_mouse_move(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    """Test that Qt mouse move events are converted and given to the canvas."""
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    press_point = (5, 10)
    with patch("scenex.model._canvas.Canvas.handle") as mock:
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
        )
    )


def test_mouse_click(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    """Test that Qt mouse click events are converted and given to the canvas."""
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    press_point = (5, 10)
    with patch("scenex.model._canvas.Canvas.handle") as mock:
        qtbot.mouseClick(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
    assert mock.call_args_list[0].args == (
        MousePressEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT,
        ),
    )
    assert mock.call_args_list[1].args == (
        MouseReleaseEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT,
        ),
    )


def test_mouse_double_click(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    """Test that Qt mouse double click events are converted and given to the canvas."""
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    press_point = (5, 10)
    with patch("scenex.model._canvas.Canvas.handle") as mock:
        # Note that in Qt a double click does NOT implicitly imply a release as well.
        qtbot.mouseDClick(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
    mock.assert_called_once_with(
        MouseDoublePressEvent(
            canvas_pos=press_point,
            world_ray=_validate_ray(evented_canvas.to_world(press_point)),
            buttons=MouseButton.LEFT,
        )
    )


def test_resize(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    """Test that Qt resize events are converted and given to the canvas."""
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    qtbot.add_widget(native)
    new_size = (400, 300)
    assert evented_canvas.width != new_size[0]
    assert evented_canvas.height != new_size[1]
    # Note that the widget must be visible for a resize event to fire
    cast("QWidget", native).setVisible(True)
    cast("QWidget", native).resize(*new_size)
    app().process_events()
    assert evented_canvas.width == new_size[0]
    assert evented_canvas.height == new_size[1]


def test_mouse_enter(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    """Test that Qt enter events are converted and given to the canvas."""
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    qtbot.add_widget(native)

    # Simulate mouse enter event by posting to event queue
    # Note that qtbot does not have a method for this
    enter_point = (0, 0)
    enter_event = QEnterEvent(
        QPointF(*enter_point),  # localPos
        QPointF(*enter_point),  # windowPos
        QPointF(*enter_point),  # screenPos
    )
    app = QApplication.instance()
    assert app is not None
    with patch("scenex.model._canvas.Canvas.handle") as mock:
        app.postEvent(native, enter_event)
        app.processEvents()

    # Verify MouseEnterEvent was passed to view filter
    mock.assert_called_once_with(
        MouseEnterEvent(
            canvas_pos=enter_point,
            world_ray=_validate_ray(evented_canvas.to_world(enter_point)),
            buttons=MouseButton.NONE,
        )
    )


def test_mouse_leave(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    """Test that Qt leave events are converted and given to the canvas."""
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    qtbot.add_widget(native)

    # NOTE: We need to first enter to establish the view as active
    enter_point = (10, 15)
    enter_event = QEnterEvent(
        QPointF(*enter_point), QPointF(*enter_point), QPointF(*enter_point)
    )
    app = QApplication.instance()
    assert app is not None
    app.postEvent(native, enter_event)
    app.processEvents()

    # Now simulate leave event
    with patch("scenex.model._canvas.Canvas.handle") as mock:
        leave_event = QEvent(QEvent.Type.Leave)
        app.postEvent(native, leave_event)

        # Process events to ensure the event is handled
        app.processEvents()
        qtbot.wait(10)

    # Verify MouseLeaveEvent was passed to view filter
    mock.assert_called_once_with(MouseLeaveEvent())


# TODO: Implement when Qt new enough
# https://doc.qt.io/qt-6/qtest.html#wheelEvent
# def test_wheel(evented_canvas: snx.Canvas):
#     pass
