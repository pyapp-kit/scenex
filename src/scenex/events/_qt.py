from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, ClassVar, cast

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtGui import QMouseEvent, QWheelEvent
from qtpy.QtWidgets import QApplication, QWidget

from scenex.events._auto import App, EventFilter
from scenex.events.events import MouseButton, MouseEvent, WheelEvent, _canvas_to_world

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from scenex import Canvas
    from scenex.events import Event


class QtEventFilter(QObject, EventFilter):
    def __init__(
        self, canvas: Any, model_canvas: Canvas, filter_func: Callable[[Event], bool]
    ) -> None:
        super(QObject, self).__init__()
        self._canvas = canvas
        self._model_canvas = model_canvas
        self._filter_func = filter_func
        self._active_button: MouseButton = MouseButton.NONE

    def eventFilter(self, a0: QObject | None = None, a1: QEvent | None = None) -> bool:
        if isinstance(a0, QWidget) and isinstance(a1, QEvent):
            if evt := self._convert_event(a1):
                return self._filter_func(evt)
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
            if not (ray := _canvas_to_world(self._model_canvas, canvas_pos)):
                return None

            etype = qevent.type()
            btn = self.mouse_btn(qevent.button())
            if etype == QEvent.Type.MouseMove:
                return MouseEvent(
                    type="move",
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=self._active_button,
                )
            elif etype == QEvent.Type.MouseButtonDblClick:
                self._active_button |= btn
                return MouseEvent(
                    type="double_click",
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=self._active_button,
                )
            elif etype == QEvent.Type.MouseButtonPress:
                self._active_button |= btn
                return MouseEvent(
                    type="press",
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=self._active_button,
                )
            elif etype == QEvent.Type.MouseButtonRelease:
                self._active_button &= ~btn
                return MouseEvent(
                    type="release",
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=self._active_button,
                )
        elif isinstance(qevent, QWheelEvent):
            # TODO: Figure out the buttons
            pos = qevent.position()
            canvas_pos = (pos.x(), pos.y())
            if not (ray := _canvas_to_world(self._model_canvas, canvas_pos)):
                return None
            return WheelEvent(
                type="wheel",
                canvas_pos=canvas_pos,
                world_ray=ray,
                buttons=self._active_button,
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

    def install_event_filter(
        self, canvas: Any, model_canvas: Canvas, filter_func: Callable[[Event], bool]
    ) -> EventFilter:
        f = QtEventFilter(canvas, model_canvas, filter_func)
        cast("QWidget", canvas).installEventFilter(f)
        return f

    def show(self, canvas: Any, visible: bool) -> None:
        cast("QWidget", canvas).setVisible(visible)
