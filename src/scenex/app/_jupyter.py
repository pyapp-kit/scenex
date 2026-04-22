from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any, cast

from app_model.types import KeyBinding, SimpleKeyBinding
from IPython import display
from jupyter_rfb import RemoteFrameBuffer

from scenex.app._auto import App, CursorType
from scenex.app._jupyter_keymap import jupyterkey2modelkey
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


class JupyterEventFilter(EventFilter):
    def __init__(
        self, widget: RemoteFrameBuffer, handler: Callable[[Event], bool]
    ) -> None:
        if not isinstance(widget, RemoteFrameBuffer):
            raise TypeError(
                f"Expected widget to be RemoteFrameBuffer, got {type(widget)}"
            )
        self._widget = widget
        self._handler = handler
        self._active_button: MouseButton = MouseButton.NONE

        self._old_event = self._widget.handle_event

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
                    filter._handler(
                        MouseMoveEvent(
                            pos=canvas_pos,
                            buttons=filter._active_button,
                        )
                    )
                elif etype == "pointer_down":
                    canvas_pos = (ev["x"], ev["y"])
                    btn = JupyterEventFilter.mouse_btn(ev["button"])
                    filter._active_button |= btn
                    filter._handler(
                        MousePressEvent(
                            pos=canvas_pos,
                            buttons=btn,
                        )
                    )
                elif etype == "double_click":
                    btn = JupyterEventFilter.mouse_btn(ev["button"])
                    canvas_pos = (ev["x"], ev["y"])
                    # FIXME: in Jupyter, a double_click event is not a pointer
                    # event. In other words, there will be no release following.
                    # This could cause unintended behavior. See
                    # https://github.com/vispy/jupyter_rfb/blob/62831dd5a87bc19b4fd5f921d802ed21141e61ec/js/lib/widget.js#L270
                    filter._handler(
                        MouseDoublePressEvent(
                            pos=canvas_pos,
                            buttons=btn,
                        )
                    )
                elif etype == "pointer_up":
                    canvas_pos = (ev["x"], ev["y"])
                    btn = JupyterEventFilter.mouse_btn(ev["button"])
                    filter._active_button &= ~btn
                    filter._handler(
                        MouseReleaseEvent(
                            pos=canvas_pos,
                            buttons=btn,
                        )
                    )
                elif etype == "pointer_enter":
                    canvas_pos = (ev["x"], ev["y"])
                    filter._active_button = MouseButton.NONE
                    if btn := ev.get("button", None):
                        filter._active_button |= JupyterEventFilter.mouse_btn(btn)
                    elif btns := ev.get("buttons", None):
                        for b in btns:
                            filter._active_button |= JupyterEventFilter.mouse_btn(b)
                    filter._handler(
                        MouseEnterEvent(
                            pos=canvas_pos,
                            buttons=filter._active_button,
                        )
                    )
                elif etype == "pointer_leave":
                    filter._handler(MouseLeaveEvent())
                elif etype == "wheel":
                    canvas_pos = (ev["x"], ev["y"])
                    filter._handler(
                        WheelEvent(
                            pos=canvas_pos,
                            buttons=filter._active_button,
                            # Note that Jupyter_rfb uses a different y convention
                            angle_delta=(ev["dx"], -ev["dy"]),
                        )
                    )
                elif etype == "key_down":
                    model_key = jupyterkey2modelkey(ev)
                    part = SimpleKeyBinding.from_int(model_key)
                    filter._handler(KeyPressEvent(key=KeyBinding(parts=[part])))
                elif etype == "key_up":
                    model_key = jupyterkey2modelkey(ev)
                    part = SimpleKeyBinding.from_int(model_key)
                    filter._handler(KeyReleaseEvent(key=KeyBinding(parts=[part])))
                elif etype == "resize":
                    filter._handler(
                        ResizeEvent(
                            width=ev["width"],
                            height=ev["height"],
                        )
                    )
                    # Note: Jupyter_rfb does a lot of stuff under the hood on resize,
                    # which we will still need to do.
                    filter._old_event(ev)

            return _handle_event

        self._widget.handle_event = MethodType(_create_handler(self), self._widget)

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
        self._widget.handle_event = self._old_event


class JupyterAppWrap(App):
    """Provider for Jupyter notebook."""

    def __init__(self) -> None:
        self._visible_canvases: set[Any] = set()

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

    def install_event_filter(
        self, widget: Any, handler: Callable[[Event], bool]
    ) -> EventFilter:
        return JupyterEventFilter(widget, handler)

    def show(self, native_widget: Any, visible: bool) -> None:
        native_canvas = cast("RemoteFrameBuffer", native_widget)
        if native_widget not in self._visible_canvases:
            self._visible_canvases.add(native_widget)
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

    def set_cursor(self, native_widget: Any, cursor: CursorType) -> None:
        # remote frame buffer exposes style via layout
        cast("RemoteFrameBuffer", native_widget).cursor = self._cursor_to_jupyter(
            cursor
        )

    def _cursor_to_jupyter(self, cursor: CursorType) -> str:
        """Convert abstract CursorType to Jupyter cursor string."""
        return {
            CursorType.DEFAULT: "default",
            CursorType.CROSS: "crosshair",
            CursorType.V_ARROW: "ns-resize",
            CursorType.H_ARROW: "ew-resize",
            CursorType.ALL_ARROW: "move",
            CursorType.BDIAG_ARROW: "nesw-resize",
            CursorType.FDIAG_ARROW: "nwse-resize",
        }[cursor]
