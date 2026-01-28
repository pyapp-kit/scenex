from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pylinalg as la
from cmap import Color
from pydantic import ConfigDict, Field, PrivateAttr
from typing_extensions import Unpack

from scenex.app.events import (
    Event,
    MouseEnterEvent,
    MouseEvent,
    MouseLeaveEvent,
    Ray,
    ResizeEvent,
)
from scenex.model._evented_list import EventedList

from ._base import EventedBase
from ._view import View  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import TypedDict

    from scenex.adaptors._base import CanvasAdaptor

    class CanvasKwargs(TypedDict, total=False):
        """TypedDict for Canvas kwargs."""

        width: int
        height: int
        background_color: Color
        visible: bool
        title: str


class Canvas(EventedBase):
    """A rendering surface that displays one or more views.

    The Canvas represents the top-level rendering context where views are displayed.
    In desktop applications, a canvas corresponds to a window. In web applications,
    it corresponds to a DOM element. Multiple views can be laid out horizontally on a
    single canvas; more complex layouts are planned in the near future.

    Attributes
    ----------
    width : int
        The width of the canvas in pixels.
    height : int
        The height of the canvas in pixels.
    background_color : Color
        The background color of the canvas.
    visible : bool
        Whether the canvas is visible. Set to True to show the canvas window.
    title : str
        The window title (desktop) or label for the canvas.

    Examples
    --------
    Create a simple canvas with default settings:
        >>> canvas = Canvas()

    Create a canvas with custom size and title:
        >>> canvas = Canvas(width=800, height=600, title="My Visualization")

    Create a canvas with multiple views side-by-side:
        >>> canvas = Canvas(width=800, height=400, views=[View(), View()])
    """

    width: int = Field(default=500, description="The width of the canvas in pixels")
    height: int = Field(default=500, description="The height of the canvas in pixels")
    background_color: Color = Field(
        default=Color("black"), description="The background color of the canvas"
    )
    visible: bool = Field(
        default=False, description="Whether the canvas window is visible"
    )
    title: str = Field(
        default="",
        description="The title displayed on the canvas window",
    )
    views: EventedList[View] = Field(default_factory=EventedList, frozen=True)

    # Private state for tracking mouse view transitions
    _last_mouse_view: View | None = PrivateAttr(default=None)

    model_config = ConfigDict(extra="forbid")

    # tell mypy and pyright that this takes children, just like Node
    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            views: Iterable[View] = (),
            **data: Unpack[CanvasKwargs],
        ) -> None: ...

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook for the model."""
        # Update all current views
        for view in self.views:
            view._canvas = self

        self.events.width.connect(self._compute_layout)
        self.events.height.connect(self._compute_layout)

        self._compute_layout()

    def close(self) -> None:
        """Close the canvas and release resources."""
        for adaptor in self._get_adaptors():
            cast("CanvasAdaptor", adaptor)._snx_close()

    def _compute_layout(self) -> None:
        total = len(self.views)
        # TODO: Support more complex layouts
        for i, view in enumerate(self.views):
            view.layout.x = (i / total) * self.width
            view.layout.y = 0
            view.layout.width = self.width / total
            view.layout.height = self.height

    @property
    def size(self) -> tuple[int, int]:
        """Return the size of the canvas."""
        return self.width, self.height

    @size.setter
    def size(self, value: tuple[int, int]) -> None:
        """Set the size of the canvas."""
        self.width, self.height = value

    def render(self) -> np.ndarray:
        """Show the canvas."""
        if adaptors := self._get_adaptors():
            return cast("CanvasAdaptor", adaptors[0])._snx_render()
        raise RuntimeError("No adaptor found for Canvas.")

    def handle(self, event: Event) -> bool:
        """Handle the passed event."""
        handled = False
        if isinstance(event, MouseEvent):
            current_view = self._containing_view(event.canvas_pos)

            # Check if we've moved between views and handle transitions
            # BEGIN UNTESTED CODE!
            # TODO: Add a test for this once multiple views are better supported
            if self._last_mouse_view != current_view:
                # Send leave event to the previous view
                if self._last_mouse_view is not None:
                    leave_event = MouseLeaveEvent()
                    self._last_mouse_view.filter_event(leave_event)

                    # Send enter event to the new view (if any)
                    if current_view is not None:
                        enter_event = MouseEnterEvent(
                            canvas_pos=event.canvas_pos,
                            world_ray=event.world_ray,
                            buttons=event.buttons,
                        )
                        current_view.filter_event(enter_event)

            self._last_mouse_view = current_view
            # END UNTESTED CODE!

            # Handle the original mouse event in the current view
            if current_view is not None:
                # Give the view a chance to observe the result
                if current_view.filter_event(event):
                    return True

                # No nodes in the view handled the event - pass it to the camera
                if current_view.camera.interactive:
                    if on_mouse := current_view.camera.controller:
                        handled |= on_mouse.handle_event(event, current_view.camera)
        elif isinstance(event, MouseLeaveEvent):
            # Mouse left the entire canvas
            if self._last_mouse_view is not None:
                handled = self._last_mouse_view.filter_event(event)
                self._last_mouse_view = None
        elif isinstance(event, ResizeEvent):
            # TODO: How might some event filter tap into the resize?
            self.size = (event.width, event.height)
        return handled

    def to_ndc(self, canvas_pos: tuple[float, float]) -> tuple[float, float] | None:
        """Map XY canvas position (pixels) to normalized device coordinates (NDC)."""
        view = self._containing_view(canvas_pos)
        if view is None:
            return None

        # Get position relative to viewport
        pos_rel = (
            canvas_pos[0] - view.layout.x,
            canvas_pos[1] - view.layout.y,
        )

        width, height = view.layout.size

        # Convert position to Normalized Device Coordinates (NDC) - i.e., within [-1, 1]
        x = pos_rel[0] / width * 2 - 1
        y = -(pos_rel[1] / height * 2 - 1)
        return (x, y)

    def to_world(self, canvas_pos: tuple[float, float]) -> Ray | None:
        """Map XY canvas position (pixels) to a Ray traveling through world space."""
        # Code adapted from:
        # https://github.com/pygfx/pygfx/pull/753/files#diff-173d643434d575e67f8c0a5bf2d7ea9791e6e03a4e7a64aa5fa2cf4172af05cdR395
        view = self._containing_view(canvas_pos)
        if view is None:
            return None

        # Convert position to Normalized Device Coordinates (NDC) - i.e., within [-1, 1]
        pos_ndc = self.to_ndc(canvas_pos)

        # Note that the camera matrix is the matrix multiplication of:
        # * The projection matrix, which projects local space (the rectangular
        #   bounds of the perspective camera) into NDC.
        # * The view matrix, i.e. the transform positioning the camera in the world.
        # The result is a matrix mapping world coordinates
        camera_matrix = view.camera.projection @ view.camera.transform.inv().T
        # Unproject the canvas NDC coordinates into world space.
        pos_world = la.vec_unproject(pos_ndc, camera_matrix)

        # To find the direction of the ray, we find a unprojected point farther away
        # and subtract the closer point.
        pos_world_farther = la.vec_unproject(pos_ndc, camera_matrix, depth=1)
        direction = pos_world_farther - pos_world
        direction = direction / np.linalg.norm(direction)

        ray = Ray(
            origin=tuple(pos_world),
            direction=tuple(direction),
            source=view,
        )
        return ray

    def _containing_view(self, pos: tuple[float, float]) -> View | None:
        for view in self.views:
            if pos in view.layout:
                return view
        return None
