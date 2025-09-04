from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, ClassVar, cast

from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtGui import QMouseEvent, QWheelEvent
from qtpy.QtWidgets import QApplication, QWidget

from scenex.app._auto import App
from scenex.app.events._events import EventFilter, MouseButton, MouseEvent, WheelEvent

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from scenex import Canvas
    from scenex.adaptors._base import CanvasAdaptor
    from scenex.app.events import Event


class QtEventFilter(QObject, EventFilter):
    def __init__(self, canvas: Any, model_canvas: Canvas) -> None:
        super().__init__()
        self._canvas = canvas
        self._model_canvas = model_canvas
        self._active_buttons: MouseButton = MouseButton.NONE

    def eventFilter(self, a0: QObject | None = None, a1: QEvent | None = None) -> bool:
        if isinstance(a0, QWidget) and isinstance(a1, QEvent):
            if evt := self._convert_event(a1):
                return self._model_canvas.handle(evt)
        return False

    def uninstall(self) -> None:
        self._canvas.removeEventFilter(self)

    def mouse_btn(self, btn: Any) -> MouseButton:
        if btn == Qt.MouseButton.LeftButton:
            return MouseButton.LEFT
        if btn == Qt.MouseButton.RightButton:
            return MouseButton.RIGHT
        if btn == Qt.MouseButton.NoButton:
            return MouseButton.NONE

        raise Exception(f"Qt mouse button {btn} is unknown")

    def _convert_event(self, qevent: QEvent) -> Event | None:
        """Convert a QEvent to a SceneX Event."""
        if isinstance(qevent, QMouseEvent):
            pos = qevent.position()
            canvas_pos = (pos.x(), pos.y())
            if not (ray := self._model_canvas.to_world(canvas_pos)):
                return None

            etype = qevent.type()
            btn = self.mouse_btn(qevent.button())
            if etype == QEvent.Type.MouseMove:
                return MouseEvent(
                    type="move",
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=self._active_buttons,
                )
            elif etype == QEvent.Type.MouseButtonDblClick:
                self._active_buttons |= btn
                return MouseEvent(
                    type="double_press",
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=btn,
                )
            elif etype == QEvent.Type.MouseButtonPress:
                self._active_buttons |= btn
                return MouseEvent(
                    type="press",
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=btn,
                )
            # FIXME user might want to know (a) which button was just released
            # and (b) which buttons are still pressed. (a) is likely more common, but we
            # may want to revise the design.
            elif etype == QEvent.Type.MouseButtonRelease:
                self._active_buttons &= ~btn
                return MouseEvent(
                    type="release",
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=btn,
                )
        elif isinstance(qevent, QWheelEvent):
            # TODO: Figure out the buttons
            pos = qevent.position()
            canvas_pos = (pos.x(), pos.y())
            if not (ray := self._model_canvas.to_world(canvas_pos)):
                return None
            return WheelEvent(
                type="wheel",
                canvas_pos=canvas_pos,
                world_ray=ray,
                buttons=self._active_buttons,
                angle_delta=(qevent.angleDelta().x(), qevent.angleDelta().y()),
            )

        return None


class QtAppWrap(App):
    """Provider for PyQt5/PySide2/PyQt6/PySide6."""

    _APP_INSTANCE: ClassVar[Any] = None
    IPY_MAGIC_KEY = "qt"

    def create_app(self) -> Any:
        if (qapp := QApplication.instance()) is None:
            # otherwise create a new QApplication
            # must be stored in a class variable to prevent garbage collection
            QtAppWrap._APP_INSTANCE = qapp = QApplication(sys.argv)
            qapp.setOrganizationName("ndv")
            qapp.setApplicationName("ndv")

        return qapp

    def run(self) -> None:
        """Run the Qt event loop."""
        app = QApplication.instance() or self.create_app()

        for wdg in QApplication.topLevelWidgets():
            wdg.raise_()

        # if ipy_shell := self._ipython_shell():
        #     # if we're already in an IPython session with %gui qt, don't block
        #     if str(ipy_shell.active_eventloop).startswith("qt"):
        #         return

        app.exec()

    def install_event_filter(self, canvas: Any, model_canvas: Canvas) -> EventFilter:
        f = QtEventFilter(canvas, model_canvas)
        cast("QWidget", canvas).installEventFilter(f)
        return f

    def show(self, canvas: CanvasAdaptor, visible: bool) -> None:
        cast("QWidget", canvas._snx_get_native()).setVisible(visible)

    def process_events(self) -> None:
        """Process events for the application."""
        QApplication.processEvents()

    def call_later(self, msec: int, func: Callable[[], None]) -> None:
        """Call `func` after `msec` milliseconds."""
        QTimer.singleShot(msec, Qt.TimerType.PreciseTimer, func)
