from __future__ import annotations

from typing import TYPE_CHECKING, Any

import wx

from scenex.events._auto import App, EventFilter
from scenex.events.events import MouseButton, MouseEvent, WheelEvent, _canvas_to_world

if TYPE_CHECKING:
    from collections.abc import Callable

    from scenex import Canvas
    from scenex.adaptors._base import CanvasAdaptor
    from scenex.events.events import Event


class WxEventFilter(EventFilter):
    def __init__(
        self,
        canvas: wx.Window,
        model_canvas: Canvas,
        filter_func: Callable[[Event], bool],
    ) -> None:
        if swdg := getattr(canvas, "_subwidget", None):
            canvas = swdg

        self._canvas = canvas
        self._model_canvas = model_canvas
        self._filter_func = filter_func
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

    def uninstall(self) -> None:
        self._canvas.Unbind(wx.EVT_LEFT_DOWN)
        self._canvas.Unbind(wx.EVT_LEFT_UP)
        self._canvas.Unbind(wx.EVT_RIGHT_DOWN)
        self._canvas.Unbind(wx.EVT_RIGHT_UP)
        self._canvas.Unbind(wx.EVT_MIDDLE_DOWN)
        self._canvas.Unbind(wx.EVT_MIDDLE_UP)
        self._canvas.Unbind(wx.EVT_MOTION)
        self._canvas.Unbind(wx.EVT_MOUSEWHEEL)

    def _on_mouse_down(self, event: wx.MouseEvent) -> None:
        btn = self._map_button(event)
        self._active_button |= btn
        pos = event.GetPosition()
        if ray := _canvas_to_world(self._model_canvas, (pos.x, pos.y)):
            self._filter_func(
                MouseEvent(
                    type="press",
                    canvas_pos=(pos.x, pos.y),
                    world_ray=ray,
                    buttons=self._active_button,
                )
            )
            event.Skip()

    def _on_mouse_up(self, event: wx.MouseEvent) -> None:
        btn = self._map_button(event)
        self._active_button &= ~btn
        pos = event.GetPosition()
        if ray := _canvas_to_world(self._model_canvas, (pos.x, pos.y)):
            self._filter_func(
                MouseEvent(
                    type="release",
                    canvas_pos=(pos.x, pos.y),
                    world_ray=ray,
                    buttons=self._active_button,
                )
            )
            event.Skip()

    def _on_mouse_move(self, event: wx.MouseEvent) -> None:
        pos = event.GetPosition()
        print(pos)
        if ray := _canvas_to_world(self._model_canvas, (pos.x, pos.y)):
            self._filter_func(
                MouseEvent(
                    type="move",
                    canvas_pos=(pos.x, pos.y),
                    world_ray=ray,
                    buttons=self._active_button,
                )
            )
            event.Skip()

    def _on_wheel(self, event: wx.MouseEvent) -> None:
        pos = event.GetPosition()
        if ray := _canvas_to_world(self._model_canvas, (pos.x, pos.y)):
            self._filter_func(
                WheelEvent(
                    type="wheel",
                    canvas_pos=(pos.x, pos.y),
                    world_ray=ray,
                    buttons=self._active_button,
                    angle_delta=(event.GetWheelRotation(), 0),
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
        filter_func: Callable[[Event], bool],
    ) -> EventFilter:
        return WxEventFilter(canvas, model_canvas, filter_func)

    def show(self, canvas: CanvasAdaptor, visible: bool) -> None:
        window = canvas._snx_get_window_ref()
        if window and window.IsOk():
            wx.CallAfter(window.Show, visible)
