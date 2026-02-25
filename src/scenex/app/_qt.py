from __future__ import annotations

import sys
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, ClassVar, cast

from qtpy.QtCore import (
    QCoreApplication,
    QEvent,
    QMetaObject,
    QObject,
    Qt,
    QThread,
    QTimer,
)
from qtpy.QtGui import QEnterEvent, QMouseEvent, QResizeEvent, QWheelEvent
from qtpy.QtWidgets import QApplication, QWidget

from scenex.app._auto import App, CursorType
from scenex.app.events import (
    EventFilter,
    MouseButton,
    MouseDoublePressEvent,
    MouseEnterEvent,
    MouseLeaveEvent,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
    ResizeEvent,
    WheelEvent,
)

# NOTE: PyQt and PySide have different names for the Slot decorator
# but they're more or less interchangeable
try:
    from qtpy.QtCore import Slot as slot  # type: ignore[attr-defined]
except ImportError:
    try:
        from qtpy.QtCore import pyqtSlot as slot
    except ImportError as e:
        raise Exception("Could not import Slot or pyqtSlot from qtpy.QtCore") from e


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from scenex import Canvas
    from scenex.adaptors._base import CanvasAdaptor
    from scenex.app._auto import P, T
    from scenex.app.events import Event


class QtEventFilter(QObject, EventFilter):
    def __init__(self, canvas: Any, model_canvas: Canvas) -> None:
        super().__init__()
        self._canvas = canvas
        self._model_canvas = model_canvas
        self._active_buttons: MouseButton = MouseButton.NONE

    def eventFilter(self, a0: QObject | None = None, a1: QEvent | None = None) -> bool:
        if isinstance(a0, QWidget) and not a0.signalsBlocked():
            if isinstance(a1, QEvent) and (evt := self._convert_event(a1)):
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
        if isinstance(qevent, QMouseEvent | QEnterEvent):
            pos = qevent.position()
            canvas_pos = (pos.x(), pos.y())
            if not (ray := self._model_canvas.to_world(canvas_pos)):
                return None

            etype = qevent.type()
            btn = self.mouse_btn(qevent.button())
            if etype == QEvent.Type.MouseMove:
                return MouseMoveEvent(
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=self._active_buttons,
                )
            elif etype == QEvent.Type.MouseButtonDblClick:
                self._active_buttons |= btn
                return MouseDoublePressEvent(
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=btn,
                )
            elif etype == QEvent.Type.MouseButtonPress:
                self._active_buttons |= btn
                return MousePressEvent(
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=btn,
                )
            elif etype == QEvent.Type.MouseButtonRelease:
                self._active_buttons &= ~btn
                return MouseReleaseEvent(
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=btn,
                )
            elif etype == QEvent.Type.Enter:
                return MouseEnterEvent(
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=self._active_buttons,
                )

        elif qevent.type() == QEvent.Type.Leave:
            return MouseLeaveEvent()

        elif isinstance(qevent, QWheelEvent):
            # TODO: Figure out the buttons
            pos = qevent.position()
            canvas_pos = (pos.x(), pos.y())
            if not (ray := self._model_canvas.to_world(canvas_pos)):
                return None
            return WheelEvent(
                canvas_pos=canvas_pos,
                world_ray=ray,
                buttons=self._active_buttons,
                angle_delta=(qevent.angleDelta().x(), qevent.angleDelta().y()),
            )

        elif isinstance(qevent, QResizeEvent):
            size = qevent.size()
            return ResizeEvent(
                width=size.width(),
                height=size.height(),
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

    def show(self, canvas: Canvas, visible: bool) -> None:
        adaptor = cast("CanvasAdaptor", canvas._get_adaptors(create=True)[0])
        cast("QWidget", adaptor._snx_get_native()).setVisible(visible)

    def process_events(self) -> None:
        """Process events for the application."""
        QApplication.processEvents()

    def call_later(self, msec: int, func: Callable[[], None]) -> None:
        """Call `func` after `msec` milliseconds."""
        QTimer.singleShot(msec, Qt.TimerType.PreciseTimer, func)

    def call_in_main_thread(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        return _call_in_main_thread(func, *args, **kwargs)

    def set_cursor(self, canvas: Canvas, cursor: CursorType) -> None:
        adaptor = cast("CanvasAdaptor", canvas._get_adaptors(create=True)[0])
        cast("QWidget", adaptor._snx_get_native()).setCursor(self._cursor_to_qt(cursor))

    def _cursor_to_qt(self, cursor: CursorType) -> Qt.CursorShape:
        """Convert abstract CursorType to Qt CursorShape."""
        return {
            CursorType.DEFAULT: Qt.CursorShape.ArrowCursor,
            CursorType.CROSS: Qt.CursorShape.CrossCursor,
            CursorType.V_ARROW: Qt.CursorShape.SizeVerCursor,
            CursorType.H_ARROW: Qt.CursorShape.SizeHorCursor,
            CursorType.ALL_ARROW: Qt.CursorShape.SizeAllCursor,
            CursorType.BDIAG_ARROW: Qt.CursorShape.SizeBDiagCursor,
            CursorType.FDIAG_ARROW: Qt.CursorShape.SizeFDiagCursor,
        }[cursor]


class MainThreadInvoker(QObject):
    _current_callable: Callable | None = None
    _moved: bool = False

    def invoke(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        """Invokes a function in the main thread and returns a Future."""
        future: Future[T] = Future()

        def wrapper() -> None:
            try:
                result = func(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        self._current_callable = wrapper
        QMetaObject.invokeMethod(
            self, "_invoke_current", Qt.ConnectionType.QueuedConnection
        )
        return future

    @slot()  # type: ignore[untyped-decorator]
    def _invoke_current(self) -> None:
        """Invokes the current callable."""
        if (cb := self._current_callable) is not None:
            cb()
            _INVOKERS.discard(self)


_INVOKERS = set()


def _call_in_main_thread(
    func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Future[T]:
    if (app := QCoreApplication.instance()) is None:
        raise RuntimeError("No Qt application instance is running")
    app_thread = app.thread()
    if QThread.currentThread() is not app_thread:
        invoker = MainThreadInvoker()
        invoker.moveToThread(app_thread)
        _INVOKERS.add(invoker)
        return invoker.invoke(func, *args, **kwargs)

    future: Future[T] = Future()
    future.set_result(func(*args, **kwargs))
    return future
