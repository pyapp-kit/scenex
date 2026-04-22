"""Tests pertaining to Qt adaptors"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from app_model.types import KeyBinding

import scenex as snx
from scenex.app import CursorType, GuiFrontend, app, determine_app
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
)
from scenex.model._transform import Transform

if TYPE_CHECKING:
    from collections.abc import Generator

    from scenex.adaptors._base import CanvasAdaptor

if determine_app() == GuiFrontend.QT:
    from qtpy.QtCore import QEvent, QPoint, QPointF, Qt
    from qtpy.QtGui import QEnterEvent, QKeyEvent, QPointingDevice
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
def evented_canvas(qtbot: QtBot) -> Generator[snx.Canvas, None, None]:
    camera = snx.Camera(transform=Transform(), interactive=True)
    scene = snx.Scene(children=[])
    view = snx.View(scene=scene, camera=camera)
    canvas = snx.Canvas(views=[view])
    native = cast(
        "CanvasAdaptor", canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    qtbot.addWidget(native)
    yield canvas
    # Cleanup
    app().process_events()


def test_mouse_press(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    press_point = (5, 10)
    # Press the left button
    qtbot.mousePress(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
    mock_filter.assert_called_once_with(
        MousePressEvent(pos=press_point, buttons=MouseButton.LEFT)
    )

    mock_filter.reset_mock()
    # Now press the right button
    qtbot.mousePress(native, Qt.MouseButton.RightButton, pos=QPoint(*press_point))
    mock_filter.assert_called_once_with(
        MousePressEvent(pos=press_point, buttons=MouseButton.RIGHT)
    )


def test_mouse_release(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    press_point = (5, 10)
    qtbot.mouseRelease(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
    mock_filter.assert_called_once_with(
        MouseReleaseEvent(pos=press_point, buttons=MouseButton.LEFT)
    )


def test_mouse_move(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    press_point = (5, 10)
    # FIXME: For some reason the mouse press is necessary for processing events?
    qtbot.mousePress(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
    qtbot.mousePress(native, Qt.MouseButton.RightButton, pos=QPoint(*press_point))
    mock_filter.reset_mock()
    qtbot.mouseMove(native, pos=QPoint(*press_point))
    mock_filter.assert_called_once_with(
        MouseMoveEvent(pos=press_point, buttons=MouseButton.LEFT | MouseButton.RIGHT)
    )


def test_mouse_click(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    press_point = (5, 10)
    qtbot.mouseClick(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))

    assert mock_filter.call_args_list[0].args == (
        MousePressEvent(pos=press_point, buttons=MouseButton.LEFT),
    )
    assert mock_filter.call_args_list[1].args == (
        MouseReleaseEvent(pos=press_point, buttons=MouseButton.LEFT),
    )


def test_mouse_double_click(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    press_point = (5, 10)
    # Note that in Qt a double click does NOT implicitly imply a release as well.
    qtbot.mouseDClick(native, Qt.MouseButton.LeftButton, pos=QPoint(*press_point))
    assert mock_filter.call_args_list[0].args == (
        MouseDoublePressEvent(pos=press_point, buttons=MouseButton.LEFT),
    )


def test_resize(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
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
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    qtbot.add_widget(native)
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    # Simulate mouse enter event by posting to event queue
    # Note that qtbot does not have a method for this
    enter_point = (0, 0)
    enter_event = QEnterEvent(
        QPointF(*enter_point),  # localPos
        QPointF(*enter_point),  # windowPos
        QPointF(*enter_point),  # screenPos
        QPointingDevice.primaryPointingDevice(),
    )
    qapp = QApplication.instance()
    assert qapp is not None
    qapp.postEvent(native, enter_event)
    qapp.processEvents()

    # Verify MouseEnterEvent was passed to Canvas.handle
    mock_filter.assert_called_once_with(
        MouseEnterEvent(pos=enter_point, buttons=MouseButton.NONE)
    )


def test_mouse_leave(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    qtbot.add_widget(native)
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    enter_point = (10, 15)
    enter_event = QEnterEvent(
        QPointF(*enter_point),
        QPointF(*enter_point),
        QPointF(*enter_point),
        QPointingDevice.primaryPointingDevice(),
    )
    qapp = QApplication.instance()
    assert qapp is not None
    # NOTE: We need to first enter to establish the view as active
    qapp.postEvent(native, enter_event)
    qapp.processEvents()
    mock_filter.reset_mock()
    # Now simulate leave event
    leave_event = QEvent(QEvent.Type.Leave)
    qapp.postEvent(native, leave_event)
    # Process events to ensure the event is handled
    qapp.processEvents()
    qtbot.wait(10)

    # Verify MouseLeaveEvent was passed to Canvas.handle
    mock_filter.assert_called_once_with(MouseLeaveEvent())


def test_key_event(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    adaptor = evented_canvas._get_adaptors(create=True)[0]
    native = cast("CanvasAdaptor", adaptor)._snx_get_native()
    qtbot.add_widget(native)
    mock_filter = MagicMock()
    evented_canvas.set_event_filter(mock_filter)

    qapp = QApplication.instance()
    assert qapp is not None
    no_mod = Qt.KeyboardModifier.NoModifier
    qapp.postEvent(native, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, no_mod))
    qapp.processEvents()

    mock_filter.assert_called_once_with(
        KeyPressEvent(key=KeyBinding.from_str("A")),
    )
    mock_filter.reset_mock()

    qapp.postEvent(native, QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_A, no_mod))
    qapp.processEvents()
    mock_filter.assert_called_once_with(
        KeyReleaseEvent(key=KeyBinding.from_str("A")),
    )


def test_set_cursor(evented_canvas: snx.Canvas, qtbot: QtBot) -> None:
    native = cast(
        "CanvasAdaptor", evented_canvas._get_adaptors(create=True)[0]
    )._snx_get_native()
    qtbot.addWidget(native)
    snx.set_cursor(evented_canvas, CursorType.CROSS)

    assert cast("QWidget", native).cursor().shape() == Qt.CursorShape.CrossCursor


# TODO: Implement when Qt new enough
# https://doc.qt.io/qt-6/qtest.html#wheelEvent
# def test_wheel(evented_canvas: snx.Canvas):
#     pass
