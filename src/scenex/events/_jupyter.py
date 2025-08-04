from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any

from IPython.display import display
from jupyter_rfb import RemoteFrameBuffer

from scenex.events._auto import App, EventFilter
from scenex.events.events import MouseButton, MouseEvent, WheelEvent, _canvas_to_world

if TYPE_CHECKING:
    from collections.abc import Callable

    from scenex import Canvas
    from scenex.events.events import Event


class JupyterEventFilter(EventFilter):
    def __init__(
        self, canvas: Any, model_canvas: Canvas, filter_func: Callable[[Event], bool]
    ) -> None:
        if not isinstance(canvas, RemoteFrameBuffer):
            raise TypeError(
                f"Expected canvas to be RemoteFrameBuffer, got {type(canvas)}"
            )
        self._canvas = canvas
        self._model_canvas = model_canvas
        self._filter_func = filter_func
        self._active_button: MouseButton = MouseButton.NONE

        self._old_event = self._canvas.handle_event
        display("Using Jupyter Event Filter")

        def _handle_event(self: RemoteFrameBuffer, ev: dict) -> None:
            display(ev)
            nonlocal model_canvas
            nonlocal filter_func
            self._active_button = MouseButton.NONE
            etype = ev["event_type"]
            if etype == "pointer_move":
                canvas_pos = (ev["x"], ev["y"])
                if world_ray := _canvas_to_world(model_canvas, canvas_pos):
                    filter_func(
                        MouseEvent(
                            type="move",
                            canvas_pos=canvas_pos,
                            world_ray=world_ray,
                            buttons=self._active_button,
                        )
                    )
            elif etype == "pointer_down":
                canvas_pos = (ev["x"], ev["y"])
                self._active_button |= JupyterEventFilter.mouse_btn(ev["button"])
                if world_ray := _canvas_to_world(model_canvas, canvas_pos):
                    filter_func(
                        MouseEvent(
                            type="press",
                            canvas_pos=canvas_pos,
                            world_ray=world_ray,
                            buttons=self._active_button,
                        )
                    )
            elif etype == "double_click":
                btn = JupyterEventFilter.mouse_btn(ev["button"])
                canvas_pos = (ev["x"], ev["y"])
                if world_ray := _canvas_to_world(model_canvas, canvas_pos):
                    # Note that in Jupyter, a double_click event is not a pointer event
                    # and as such, we need to handle both press and release. See
                    # https://github.com/vispy/jupyter_rfb/blob/62831dd5a87bc19b4fd5f921d802ed21141e61ec/js/lib/widget.js#L270
                    filter_func(
                        MouseEvent(
                            type="press",
                            canvas_pos=canvas_pos,
                            world_ray=world_ray,
                            buttons=btn,
                        )
                    )
                    # Release
                    filter_func(
                        MouseEvent(
                            type="release",
                            canvas_pos=canvas_pos,
                            world_ray=world_ray,
                            buttons=btn,
                        )
                    )
            elif etype == "pointer_up":
                canvas_pos = (ev["x"], ev["y"])
                self._active_button |= JupyterEventFilter.mouse_btn(ev["button"])
                if world_ray := _canvas_to_world(model_canvas, canvas_pos):
                    filter_func(
                        MouseEvent(
                            type="release",
                            canvas_pos=canvas_pos,
                            world_ray=world_ray,
                            buttons=self._active_button,
                        )
                    )
            # elif etype == "wheel":
            # if not intercepted:
            #     self._old_event(ev)

        self._canvas.handle_event = MethodType(_handle_event, self._canvas)

    @classmethod
    def mouse_btn(cls, btn: Any) -> MouseButton:
        if btn == 0:
            return MouseButton.NONE
        if btn == 1:
            return MouseButton.LEFT
        if btn == 2:
            return MouseButton.RIGHT
        if btn == 3:
            return MouseButton.MIDDLE

        raise Exception(f"Jupyter mouse button {btn} is unknown")

    def uninstall(self) -> None:
        self._canvas.handle_event = self._old_event

    def _on_wheel(self, event: dict) -> None:
        pos = (event["x"], event["y"])
        ray = _canvas_to_world(self._model_canvas, pos)
        self._filter_func(
            WheelEvent(
                type="wheel",
                canvas_pos=pos,
                world_ray=ray,
                buttons=self._active_button,
                angle_delta=(event["delta_x"], event["delta_y"]),
            )
        )


class JupyterAppWrap(App):
    """Provider for Jupyter notebook."""

    def create_app(self) -> Any:
        # No explicit app needed for Jupyter
        return None

    def install_event_filter(
        self, canvas: Any, model_canvas: Canvas, filter_func: Callable[[Event], bool]
    ) -> EventFilter:
        return JupyterEventFilter(canvas, model_canvas, filter_func)

    def show(self, canvas: Any, visible: bool) -> None:
        if visible:
            display("Showing!")
            display(canvas)
