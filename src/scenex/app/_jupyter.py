from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any, cast

from IPython import display
from jupyter_rfb import RemoteFrameBuffer

from scenex.app._auto import App
from scenex.app.events._events import (
    EventFilter,
    MouseButton,
    MouseDoublePressEvent,
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


class JupyterEventFilter(EventFilter):
    def __init__(self, canvas: Any, model_canvas: Canvas) -> None:
        if not isinstance(canvas, RemoteFrameBuffer):
            raise TypeError(
                f"Expected canvas to be RemoteFrameBuffer, got {type(canvas)}"
            )
        self._canvas = canvas
        self._model_canvas = model_canvas
        self._active_button: MouseButton = MouseButton.NONE

        self._old_event = self._canvas.handle_event

        def _create_handler(
            filter: JupyterEventFilter,
        ) -> Callable[[RemoteFrameBuffer, dict], None]:
            def _handle_event(self: RemoteFrameBuffer, ev: dict) -> None:
                etype = ev["event_type"]
                if etype == "pointer_move":
                    filter._active_button = MouseButton.NONE
                    if btn := ev.get("button", None):
                        filter._active_button |= JupyterEventFilter.mouse_btn(btn)
                    elif btns := ev.get("buttons", None):
                        for b in btns:
                            filter._active_button |= JupyterEventFilter.mouse_btn(b)
                    canvas_pos = (ev["x"], ev["y"])
                    if world_ray := filter._model_canvas.to_world(canvas_pos):
                        filter._model_canvas.handle(
                            MouseMoveEvent(
                                canvas_pos=canvas_pos,
                                world_ray=world_ray,
                                buttons=filter._active_button,
                            )
                        )
                elif etype == "pointer_down":
                    canvas_pos = (ev["x"], ev["y"])
                    btn = JupyterEventFilter.mouse_btn(ev["button"])
                    filter._active_button |= btn
                    if world_ray := filter._model_canvas.to_world(canvas_pos):
                        filter._model_canvas.handle(
                            MousePressEvent(
                                canvas_pos=canvas_pos,
                                world_ray=world_ray,
                                buttons=btn,
                            )
                        )
                elif etype == "double_click":
                    btn = JupyterEventFilter.mouse_btn(ev["button"])
                    canvas_pos = (ev["x"], ev["y"])
                    if world_ray := filter._model_canvas.to_world(canvas_pos):
                        # FIXME: in Jupyter, a double_click event is not a pointer
                        # event. In other words, there will be no release following.
                        # This could cause unintended behavior. See
                        # https://github.com/vispy/jupyter_rfb/blob/62831dd5a87bc19b4fd5f921d802ed21141e61ec/js/lib/widget.js#L270
                        filter._model_canvas.handle(
                            MouseDoublePressEvent(
                                canvas_pos=canvas_pos,
                                world_ray=world_ray,
                                buttons=btn,
                            )
                        )
                elif etype == "pointer_up":
                    canvas_pos = (ev["x"], ev["y"])
                    btn = JupyterEventFilter.mouse_btn(ev["button"])
                    filter._active_button &= ~btn
                    if world_ray := filter._model_canvas.to_world(canvas_pos):
                        filter._model_canvas.handle(
                            MouseReleaseEvent(
                                canvas_pos=canvas_pos,
                                world_ray=world_ray,
                                buttons=btn,
                            )
                        )
                elif etype == "wheel":
                    canvas_pos = (ev["x"], ev["y"])
                    if world_ray := filter._model_canvas.to_world(canvas_pos):
                        filter._model_canvas.handle(
                            WheelEvent(
                                canvas_pos=canvas_pos,
                                world_ray=world_ray,
                                buttons=filter._active_button,
                                # Note that Jupyter_rfb uses a different y convention
                                angle_delta=(ev["dx"], -ev["dy"]),
                            )
                        )
                elif etype == "resize":
                    filter._model_canvas.handle(
                        ResizeEvent(
                            width=ev["width"],
                            height=ev["height"],
                        )
                    )

            return _handle_event

        self._canvas.handle_event = MethodType(_create_handler(self), self._canvas)

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


class JupyterAppWrap(App):
    """Provider for Jupyter notebook."""

    def __init__(self) -> None:
        self._visible_canvases: set[CanvasAdaptor] = set()

    # def is_running(self) -> bool:
    #     if ipy_shell := self._ipython_shell():
    #         return bool(ipy_shell.__class__.__name__ == "ZMQInteractiveShell")
    #     return False

    def create_app(self) -> Any:
        # if not self.is_running() and not os.getenv("PYTEST_CURRENT_TEST"):
        #     # if we got here, it probably means that someone used
        #     # NDV_GUI_FRONTEND=jupyter without actually being in a jupyter notebook
        #     # we allow it in tests, but not in normal usage.
        #     raise RuntimeError(  # pragma: no cover
        #         "Jupyter is not running a notebook shell.  Cannot create app."
        #     )

        # No app creation needed...
        # but make sure we can actually import the stuff we need
        import ipywidgets  # noqa: F401
        import jupyter  # noqa: F401

    def run(self) -> None:
        """Run the application."""
        # No explicit run method needed for Jupyter
        pass

    def install_event_filter(self, canvas: Any, model_canvas: Canvas) -> EventFilter:
        return JupyterEventFilter(canvas, model_canvas)

    def show(self, canvas: CanvasAdaptor, visible: bool) -> None:
        native_canvas = cast("RemoteFrameBuffer", canvas._snx_get_native())
        if canvas not in self._visible_canvases:
            self._visible_canvases.add(canvas)
            display.display(native_canvas)
        native_canvas.layout.display = "flex" if visible else "none"

    def process_events(self) -> None:
        """Process events for the application."""
        pass

    def call_later(self, msec: int, func: Callable[[], None]) -> None:
        """Call `func` after `msec` milliseconds."""
        # generic implementation using python threading

        from threading import Timer

        Timer(msec / 1000, func).start()
