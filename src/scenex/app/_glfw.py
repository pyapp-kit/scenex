from __future__ import annotations

from typing import TYPE_CHECKING

import glfw

from scenex.app._auto import App
from scenex.app.events._events import EventFilter, MouseButton, MouseEvent, WheelEvent

if TYPE_CHECKING:
    from typing import Any

    from scenex import Canvas
    from scenex.adaptors._base import CanvasAdaptor

BUTTONMAP = {
    glfw.MOUSE_BUTTON_LEFT: MouseButton.LEFT,
    glfw.MOUSE_BUTTON_RIGHT: MouseButton.RIGHT,
    glfw.MOUSE_BUTTON_MIDDLE: MouseButton.MIDDLE,
}


class GlfwEventFilter(EventFilter):
    def __init__(self, canvas: Any, model_canvas: Canvas) -> None:
        self._canvas = model_canvas
        self._active_button: MouseButton = MouseButton.NONE
        self._window_id = self._guess_id(canvas)
        # TODO: Maybe save the old callbacks?
        glfw.set_cursor_pos_callback(self._window_id, self._cursor_pos_callback)
        glfw.set_cursor_enter_callback(
            self._window_id, self._cursor_enter_leave_callback
        )
        glfw.set_mouse_button_callback(self._window_id, self._mouse_button_callback)
        glfw.set_scroll_callback(self._window_id, self._mouse_scroll_callback)
        self.pos = (0, 0)

    def _guess_id(self, canvas: Any) -> Any:
        # vispy
        if window := getattr(canvas, "_id", None):
            return window
        # rendercanvas
        if window := getattr(canvas, "_window", None):
            return window

    def uninstall(self) -> None:
        raise NotImplementedError(
            "Uninstalling GLFW event filters is not yet supported."
        )

    def _cursor_pos_callback(self, window: Any, xpos: float, ypos: float) -> None:
        """Handle cursor position events."""
        canvas_pos = (xpos, ypos)
        if ray := self._canvas.to_world(canvas_pos):
            self._canvas.handle(
                MouseEvent(
                    type="move",
                    canvas_pos=canvas_pos,
                    world_ray=ray,
                    buttons=self._active_button,
                )
            )

    def _cursor_enter_leave_callback(self, window: Any, entered: int) -> None:
        """Handle enter/leave events."""
        if entered:
            # entered window
            pass
        else:
            # left window
            pass

    def _mouse_button_callback(
        self, window: Any, button: int, action: int, mods: int
    ) -> None:
        pos = glfw.get_cursor_pos(window)
        if not (ray := self._canvas.to_world(pos)):
            return

        # Mouse click event
        if button in BUTTONMAP:
            if action == glfw.PRESS:
                self._active_button |= BUTTONMAP[button]
                self._canvas.handle(
                    MouseEvent(
                        type="press",
                        canvas_pos=pos,
                        world_ray=ray,
                        buttons=self._active_button,
                    )
                )
            elif action == glfw.RELEASE:
                self._active_button &= ~BUTTONMAP[button]
                self._canvas.handle(
                    MouseEvent(
                        type="release",
                        canvas_pos=pos,
                        world_ray=ray,
                        buttons=self._active_button,
                    )
                )

    def _mouse_scroll_callback(
        self, window: Any, xoffset: float, yoffset: float
    ) -> None:
        pos = glfw.get_cursor_pos(window)
        if not (ray := self._canvas.to_world(pos)):
            return

        # Mouse wheel event
        self._canvas.handle(
            WheelEvent(
                type="scroll",
                canvas_pos=pos,
                world_ray=ray,
                buttons=self._active_button,
                # Rendercanvas uses 100x and that works nice :)
                angle_delta=(xoffset * 100, yoffset * 100),
            )
        )


class GlfwAppWrap(App):
    """Provider for GLFW."""

    def create_app(self) -> Any:
        glfw.init()
        # Nothing really to return here...
        return None

    def run(self) -> None:
        """Run the GLFW application."""
        # TODO
        raise NotImplementedError(
            "GLFW event loop not implemented yet. In the meantime, "
            "Uninstall GLFW and run another canvas framework."
        )

    def install_event_filter(self, canvas: Any, model_canvas: Canvas) -> EventFilter:
        return GlfwEventFilter(canvas, model_canvas)

    def show(self, canvas: CanvasAdaptor, visible: bool) -> None:
        if visible:
            glfw.show_window(canvas._snx_get_native())
        else:
            glfw.hide_window(canvas._snx_get_native())
