from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, cast

import wx

from scenex.app._auto import App, CursorType
from scenex.app.events._events import (
    EventFilter,
    MouseButton,
    MouseEnterEvent,
    MouseLeaveEvent,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
    ResizeEvent,
    WheelEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from scenex import Canvas
    from scenex.adaptors._base import CanvasAdaptor
    from scenex.app._auto import P, T


class WxEventFilter(EventFilter):
    def __init__(
        self,
        canvas: wx.Window,
        model_canvas: Canvas,
    ) -> None:
        self._canvas = canvas
        self._model_canvas = model_canvas
        self._active_button: MouseButton = MouseButton.NONE
        self._install_events()

    def _install_events(self) -> None:
        self._canvas.Bind(wx.EVT_LEFT_DOWN, handler=self._on_mouse_down)
        self._canvas.Bind(wx.EVT_LEFT_UP, handler=self._on_mouse_up)
        self._canvas.Bind(wx.EVT_RIGHT_DOWN, handler=self._on_mouse_down)
        self._canvas.Bind(wx.EVT_RIGHT_UP, handler=self._on_mouse_up)
        self._canvas.Bind(wx.EVT_MIDDLE_DOWN, handler=self._on_mouse_down)
        self._canvas.Bind(wx.EVT_MIDDLE_UP, handler=self._on_mouse_up)
        self._canvas.Bind(wx.EVT_MOTION, handler=self._on_mouse_move)
        self._canvas.Bind(wx.EVT_MOUSEWHEEL, handler=self._on_wheel)
        self._canvas.Bind(wx.EVT_LEAVE_WINDOW, handler=self._on_leave_window)
        self._canvas.Bind(wx.EVT_ENTER_WINDOW, handler=self._on_enter_window)
        self._canvas.Bind(wx.EVT_SIZE, handler=self._on_resize)

    def uninstall(self) -> None:
        self._canvas.Unbind(wx.EVT_LEFT_DOWN)
        self._canvas.Unbind(wx.EVT_LEFT_UP)
        self._canvas.Unbind(wx.EVT_RIGHT_DOWN)
        self._canvas.Unbind(wx.EVT_RIGHT_UP)
        self._canvas.Unbind(wx.EVT_MIDDLE_DOWN)
        self._canvas.Unbind(wx.EVT_MIDDLE_UP)
        self._canvas.Unbind(wx.EVT_MOTION)
        self._canvas.Unbind(wx.EVT_MOUSEWHEEL)
        self._canvas.Unbind(wx.EVT_SIZE)

    def _on_leave_window(self, event: wx.MouseEvent) -> None:
        self._model_canvas.handle(MouseLeaveEvent())
        event.Skip()

    def _on_enter_window(self, event: wx.MouseEvent) -> None:
        pos = event.GetPosition()
        if ray := self._model_canvas.to_world((pos.x, pos.y)):
            self._model_canvas.handle(
                MouseEnterEvent(
                    canvas_pos=(pos.x, pos.y),
                    world_ray=ray,
                    buttons=self._active_button,
                )
            )
            event.Skip()

    def _on_resize(self, event: wx.SizeEvent) -> None:
        self._model_canvas.handle(
            ResizeEvent(
                width=event.GetSize().GetWidth(),
                height=event.GetSize().GetHeight(),
            )
        )
        event.Skip()

    def _on_mouse_down(self, event: wx.MouseEvent) -> None:
        btn = self._map_button(event)
        self._active_button |= btn
        pos = event.GetPosition()
        if ray := self._model_canvas.to_world((pos.x, pos.y)):
            self._model_canvas.handle(
                MousePressEvent(canvas_pos=(pos.x, pos.y), world_ray=ray, buttons=btn)
            )
            event.Skip()

    def _on_mouse_up(self, event: wx.MouseEvent) -> None:
        btn = self._map_button(event)
        self._active_button &= ~btn
        pos = event.GetPosition()
        if ray := self._model_canvas.to_world((pos.x, pos.y)):
            self._model_canvas.handle(
                MouseReleaseEvent(
                    canvas_pos=(pos.x, pos.y),
                    world_ray=ray,
                    buttons=btn,
                )
            )
            event.Skip()

    def _on_mouse_move(self, event: wx.MouseEvent) -> None:
        pos = event.GetPosition()
        if ray := self._model_canvas.to_world((pos.x, pos.y)):
            self._model_canvas.handle(
                MouseMoveEvent(
                    canvas_pos=(pos.x, pos.y),
                    world_ray=ray,
                    buttons=self._active_button,
                )
            )
            event.Skip()

    def _on_wheel(self, event: wx.MouseEvent) -> None:
        pos = event.GetPosition()
        if ray := self._model_canvas.to_world((pos.x, pos.y)):
            if event.GetWheelAxis() == 0:
                # Vertical Scroll
                angle_delta = (0, event.GetWheelRotation())
            else:
                # Horizontal Scroll
                angle_delta = (event.GetWheelRotation(), 0)

            self._model_canvas.handle(
                WheelEvent(
                    canvas_pos=(pos.x, pos.y),
                    world_ray=ray,
                    buttons=self._active_button,
                    angle_delta=angle_delta,
                )
            )
            event.Skip()

    def _map_button(self, event: wx.MouseEvent) -> MouseButton:
        if event.LeftDown() or event.LeftUp():
            return MouseButton.LEFT
        if event.RightDown() or event.RightUp():
            return MouseButton.RIGHT
        if event.MiddleDown() or event.MiddleUp():
            return MouseButton.MIDDLE
        return MouseButton.NONE


class WxAppWrap(App):
    """Provider for wxPython."""

    def create_app(self) -> Any:
        if wx.App.Get():
            return wx.App.Get()
        return wx.App(False)

    def run(self) -> None:
        app = wx.App.Get() or self.create_app()

        # if ipy_shell := self._ipython_shell():
        #     # if we're already in an IPython session with %gui qt, don't block
        #     if str(ipy_shell.active_eventloop).startswith("wx"):
        #         return

        app.MainLoop()

    def install_event_filter(
        self,
        canvas: wx.Window,
        model_canvas: Canvas,
    ) -> EventFilter:
        return WxEventFilter(canvas, model_canvas)

    def show(self, canvas: CanvasAdaptor, visible: bool) -> None:
        canvas._snx_get_native().Show(visible)
        self.process_events()

    def process_events(self) -> None:
        """Process events."""
        wx.SafeYield()

    def call_later(self, msec: int, func: Callable[[], None]) -> None:
        """Call `func` after `msec` milliseconds."""
        wx.CallLater(msec, func)

    def call_in_main_thread(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        return call_in_main_thread(func, *args, **kwargs)

    def set_cursor(self, canvas: Canvas, cursor: CursorType) -> None:
        adaptor = cast("CanvasAdaptor", canvas._get_adaptors(create=True)[0])
        native = cast("wx.Window", adaptor._snx_get_native())
        # wx Cursor objects are immutable; just set a new one
        native.SetCursor(self._cursor_to_wx(cursor))

    def _cursor_to_wx(self, cursor: CursorType) -> wx.Cursor:
        """Convert abstract CursorType to wx.Cursor."""
        return {
            CursorType.DEFAULT: wx.Cursor(wx.CURSOR_ARROW),
            CursorType.CROSS: wx.Cursor(wx.CURSOR_CROSS),
            CursorType.V_ARROW: wx.Cursor(wx.CURSOR_SIZENS),
            CursorType.H_ARROW: wx.Cursor(wx.CURSOR_SIZEWE),
            CursorType.ALL_ARROW: wx.Cursor(wx.CURSOR_SIZING),
            CursorType.BDIAG_ARROW: wx.Cursor(wx.CURSOR_SIZENESW),
            CursorType.FDIAG_ARROW: wx.Cursor(wx.CURSOR_SIZENWSE),
        }[cursor]


class MainThreadInvoker:
    def __init__(self) -> None:
        """Utility for invoking functions in the main thread."""
        # Ensure this is initialized from the main thread
        if not wx.IsMainThread():  # pyright: ignore[reportCallIssue]
            raise RuntimeError(
                "MainThreadInvoker must be initialized in the main thread"
            )

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

        wx.CallAfter(wrapper)
        return future


_MAIN_THREAD_INVOKER = MainThreadInvoker()


def call_in_main_thread(
    func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Future[T]:
    if not wx.IsMainThread():  # pyright: ignore[reportCallIssue]
        return _MAIN_THREAD_INVOKER.invoke(func, *args, **kwargs)

    future: Future[T] = Future()
    future.set_result(func(*args, **kwargs))
    return future
