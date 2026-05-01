from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, cast

import wx
from app_model.types import KeyBinding, SimpleKeyBinding

from scenex.app._auto import App, CursorType
from scenex.app._wx_keymap import wxevent2modelkey
from scenex.app.events._events import (
    Event,
    EventFilter,
    KeyPressEvent,
    KeyReleaseEvent,
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

if TYPE_CHECKING:
    from collections.abc import Callable

    from scenex.app._auto import P, T


class WxEventFilter(EventFilter):
    def __init__(
        self,
        widget: wx.Window,
        handler: Callable[[Event], bool],
    ) -> None:
        self._widget = widget
        self._handler = handler
        self._install_events()

    def _install_events(self) -> None:
        self._widget.Bind(wx.EVT_LEFT_DOWN, handler=self._on_mouse_down)
        self._widget.Bind(wx.EVT_LEFT_UP, handler=self._on_mouse_up)
        self._widget.Bind(wx.EVT_RIGHT_DOWN, handler=self._on_mouse_down)
        self._widget.Bind(wx.EVT_RIGHT_UP, handler=self._on_mouse_up)
        self._widget.Bind(wx.EVT_MIDDLE_DOWN, handler=self._on_mouse_down)
        self._widget.Bind(wx.EVT_MIDDLE_UP, handler=self._on_mouse_up)
        self._widget.Bind(wx.EVT_LEFT_DCLICK, handler=self._on_left_dclick)
        self._widget.Bind(wx.EVT_RIGHT_DCLICK, handler=self._on_right_dclick)
        self._widget.Bind(wx.EVT_MIDDLE_DCLICK, handler=self._on_middle_dclick)
        self._widget.Bind(wx.EVT_MOTION, handler=self._on_mouse_move)
        self._widget.Bind(wx.EVT_MOUSEWHEEL, handler=self._on_wheel)
        self._widget.Bind(wx.EVT_LEAVE_WINDOW, handler=self._on_leave_window)
        self._widget.Bind(wx.EVT_ENTER_WINDOW, handler=self._on_enter_window)
        self._widget.Bind(wx.EVT_SIZE, handler=self._on_resize)
        self._widget.Bind(wx.EVT_KEY_DOWN, handler=self._on_key_down)
        self._widget.Bind(wx.EVT_KEY_UP, handler=self._on_key_up)

    def uninstall(self) -> None:
        self._widget.Unbind(wx.EVT_LEFT_DOWN)
        self._widget.Unbind(wx.EVT_LEFT_UP)
        self._widget.Unbind(wx.EVT_RIGHT_DOWN)
        self._widget.Unbind(wx.EVT_RIGHT_UP)
        self._widget.Unbind(wx.EVT_MIDDLE_DOWN)
        self._widget.Unbind(wx.EVT_MIDDLE_UP)
        self._widget.Unbind(wx.EVT_LEFT_DCLICK)
        self._widget.Unbind(wx.EVT_RIGHT_DCLICK)
        self._widget.Unbind(wx.EVT_MIDDLE_DCLICK)
        self._widget.Unbind(wx.EVT_MOTION)
        self._widget.Unbind(wx.EVT_MOUSEWHEEL)
        self._widget.Unbind(wx.EVT_LEAVE_WINDOW)
        self._widget.Unbind(wx.EVT_ENTER_WINDOW)
        self._widget.Unbind(wx.EVT_SIZE)
        self._widget.Unbind(wx.EVT_KEY_DOWN)
        self._widget.Unbind(wx.EVT_KEY_UP)

    def _on_leave_window(self, event: wx.MouseEvent) -> None:
        self._handler(MouseLeaveEvent())
        event.Skip()

    def _on_enter_window(self, event: wx.MouseEvent) -> None:
        pos = event.GetPosition()
        self._handler(
            MouseEnterEvent(
                pos=(pos.x, pos.y),
                buttons=self._get_active_buttons(event),
            )
        )
        event.Skip()

    def _on_resize(self, event: wx.SizeEvent) -> None:
        self._handler(
            ResizeEvent(
                width=event.GetSize().GetWidth(),
                height=event.GetSize().GetHeight(),
            )
        )
        event.Skip()

    def _on_left_dclick(self, event: wx.MouseEvent) -> None:
        # NOTE that wx does not provide the button in the double click event
        # so we need a separate handler for each button type to know which one
        # was double-clicked
        pos = event.GetPosition()
        self._handler(
            MouseDoublePressEvent(pos=(pos.x, pos.y), buttons=MouseButton.LEFT)
        )
        event.Skip()

    def _on_right_dclick(self, event: wx.MouseEvent) -> None:
        # NOTE that wx does not provide the button in the double click event
        # so we need a separate handler for each button type to know which one
        # was double-clicked
        pos = event.GetPosition()
        self._handler(
            MouseDoublePressEvent(pos=(pos.x, pos.y), buttons=MouseButton.RIGHT)
        )
        event.Skip()

    def _on_middle_dclick(self, event: wx.MouseEvent) -> None:
        # NOTE that wx does not provide the button in the double click event
        # so we need a separate handler for each button type to know which one
        # was double-clicked
        pos = event.GetPosition()
        self._handler(
            MouseDoublePressEvent(pos=(pos.x, pos.y), buttons=MouseButton.MIDDLE)
        )
        event.Skip()

    def _on_mouse_down(self, event: wx.MouseEvent) -> None:
        # Find only the NEW button being pressed
        btn = self._get_pressed_button(event)
        pos = event.GetPosition()
        self._handler(MousePressEvent(pos=(pos.x, pos.y), buttons=btn))
        event.Skip()

    def _on_mouse_up(self, event: wx.MouseEvent) -> None:
        btn = self._get_released_button(event)
        pos = event.GetPosition()
        self._handler(MouseReleaseEvent(pos=(pos.x, pos.y), buttons=btn))
        event.Skip()

    def _on_mouse_move(self, event: wx.MouseEvent) -> None:
        pos = event.GetPosition()
        self._handler(
            MouseMoveEvent(
                pos=(pos.x, pos.y),
                buttons=self._get_active_buttons(event),
            )
        )
        event.Skip()

    def _on_wheel(self, event: wx.MouseEvent) -> None:
        pos = event.GetPosition()
        if event.GetWheelAxis() == 0:
            # Vertical Scroll
            angle_delta = (0, event.GetWheelRotation())
        else:
            # Horizontal Scroll
            angle_delta = (event.GetWheelRotation(), 0)

        self._handler(
            WheelEvent(
                pos=(pos.x, pos.y),
                buttons=self._get_active_buttons(event),
                angle_delta=angle_delta,
            )
        )
        event.Skip()

    def _on_key_down(self, event: wx.KeyEvent) -> None:
        model_key = wxevent2modelkey(event)
        part = SimpleKeyBinding.from_int(model_key)
        self._handler(KeyPressEvent(key=KeyBinding(parts=[part])))
        event.Skip()

    def _on_key_up(self, event: wx.KeyEvent) -> None:
        model_key = wxevent2modelkey(event)
        part = SimpleKeyBinding.from_int(model_key)
        self._handler(KeyReleaseEvent(key=KeyBinding(parts=[part])))
        event.Skip()

    def _get_active_buttons(self, event: wx.MouseEvent) -> MouseButton:
        """Map a DOWN wx.MouseEvent to a MouseButton."""
        button = MouseButton.NONE
        if event.LeftIsDown():
            button |= MouseButton.LEFT
        if event.RightIsDown():
            button |= MouseButton.RIGHT
        if event.MiddleIsDown():
            button |= MouseButton.MIDDLE
        return button

    def _get_pressed_button(self, event: wx.MouseEvent) -> MouseButton:
        """Map an UP wx.MouseEvent to a MouseButton."""
        button = MouseButton.NONE
        if event.LeftDown():
            button |= MouseButton.LEFT
        if event.RightDown():
            button |= MouseButton.RIGHT
        if event.MiddleDown():
            button |= MouseButton.MIDDLE
        return button

    def _get_released_button(self, event: wx.MouseEvent) -> MouseButton:
        """Map an MOVE wx.MouseEvent to a MouseButton."""
        button = MouseButton.NONE
        if event.LeftUp():
            button |= MouseButton.LEFT
        if event.RightUp():
            button |= MouseButton.RIGHT
        if event.MiddleUp():
            button |= MouseButton.MIDDLE
        return button


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
        widget: wx.Window,
        handler: Callable[[Event], bool],
    ) -> EventFilter:
        return WxEventFilter(widget, handler)

    def show(self, native_widget: Any, visible: bool) -> None:
        native_widget.Show(visible)
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

    def set_cursor(self, native_widget: Any, cursor: CursorType) -> None:
        # wx Cursor objects are immutable; just set a new one
        cast("wx.Window", native_widget).SetCursor(self._cursor_to_wx(cursor))

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
