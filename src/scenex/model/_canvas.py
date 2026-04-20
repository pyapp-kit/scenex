from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from cmap import Color
from pydantic import ConfigDict, Field, PrivateAttr
from typing_extensions import Unpack

from scenex.app.events import (
    Event,
    MouseEnterEvent,
    MouseEvent,
    MouseLeaveEvent,
    ResizeEvent,
)
from scenex.model._evented_list import EventedList

from ._base import EventedBase
from ._view import View  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy as np
    from typing_extensions import TypedDict

    from scenex.adaptors._base import CanvasAdaptor

    class CanvasKwargs(TypedDict, total=False):
        """TypedDict for Canvas kwargs."""

        width: int
        height: int
        background_color: Color
        visible: bool
        title: str


logger = logging.getLogger(__name__)


class Canvas(EventedBase):
    """A rendering surface that displays one or more views.

    The Canvas represents the top-level rendering context where views are displayed.
    In desktop applications, a canvas corresponds to a window. In web applications,
    it corresponds to a DOM element. Multiple views can be arranged on a single canvas
    using their layout parameters.

    Examples
    --------
    Create a simple canvas with default settings:
        >>> canvas = Canvas()

    Create a canvas with custom size and title:
        >>> canvas = Canvas(width=800, height=600, title="My Visualization")

    Create a canvas with multiple views side-by-side:
        >>> canvas = Canvas(width=800, height=400, views=[View(), View()])
    """

    width: int = Field(default=600, description="The width of the canvas in pixels")
    height: int = Field(default=600, description="The height of the canvas in pixels")
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
    views: EventedList[View] = Field(
        default_factory=EventedList,
        # Prevent reassigning this field - we'd lose our signal connections
        frozen=True,
    )

    # Private state for tracking mouse view transitions
    _last_mouse_view: View | None = PrivateAttr(default=None)
    _filter: Callable[[Event], bool] | None = PrivateAttr(default=None)

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
            view.canvas = self

        self.views.item_inserted.connect(self._on_view_inserted)
        self.views.item_removed.connect(self._on_view_removed)
        self.views.item_changed.connect(self._on_view_changed)

    def close(self) -> None:
        """Close the canvas and release resources."""
        for adaptor in self._get_adaptors():
            cast("CanvasAdaptor", adaptor)._snx_close()

    def rect_for(self, view: View) -> tuple[int, int, int, int]:
        """The pixel rect (x, y, width, height) for a view, computed from its layout."""
        x = view.layout.x_start.resolve(self.width)
        w = view.layout.x_end.resolve(self.width) - x
        y = view.layout.y_start.resolve(self.height)
        h = view.layout.y_end.resolve(self.height) - y
        return (x, y, w, h)

    def content_rect_for(self, view: View) -> tuple[int, int, int, int]:
        """The pixel rect (x, y, width, height) of the content area for a view.

        Applies the view's padding, border_width, and margin insets to the
        outer rect returned by ``rect_for``.
        """
        x, y, w, h = self.rect_for(view)
        layout = view.layout
        offset = int(layout.padding + layout.border_width + layout.margin)
        return (x + offset, y + offset, w - 2 * offset, h - 2 * offset)

    def _on_view_inserted(self, idx: int, view: View) -> None:
        # Set canvas reference to this if it isn't set
        if view.canvas is not self:
            view.canvas = self

    def _on_view_removed(self, idx: int, view: View) -> None:
        # Unset canvas reference to this if it is still set
        if view.canvas is self:
            view.canvas = None

    def _on_view_changed(
        self,
        idx: int | slice,
        old_views: View | Sequence[View],
        new_views: View | Sequence[View],
    ) -> None:
        if not isinstance(old_views, Sequence):
            old_views = [old_views]
        for view in old_views:
            if view.canvas is self:
                view.canvas = None

        if not isinstance(new_views, Sequence):
            new_views = [new_views]
        for view in new_views:
            if view.canvas is not self:
                view.canvas = self

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

    def set_event_filter(
        self, callable: Callable[[Event], bool] | None
    ) -> Callable[[Event], bool] | None:
        """Register a callable to filter all canvas events before view dispatch.

        Parameters
        ----------
        callable : Callable[[Event], bool] | None
            A callable that takes an Event and returns True if the event was handled
            and should not be propagated further, False otherwise. Pass None to remove
            any existing filter.

        Returns
        -------
        Callable[[Event], bool] | None
            The previous event filter, or None if there was no filter.
        """
        old, self._filter = self._filter, callable
        return old

    def filter_event(self, event: Event) -> bool:
        """Pass *event* through the canvas-level filter, if any.

        Returns True iff the event was handled and should not propagate.
        """
        if self._filter:
            handled = self._filter(event)
            if not isinstance(handled, bool):
                logger.warning(
                    f"Canvas event filter {self._filter} did not return a boolean. "
                    "Returning False."
                )
                handled = False
            return handled
        return False

    def handle(self, event: Event) -> bool:
        """Handle the passed event."""
        # 0. Handle events pertaining to the canvas model
        if isinstance(event, ResizeEvent):
            self.size = (event.width, event.height)

        # 1. Canvas-level filter sees all events first.
        if self.filter_event(event):
            return True

        # 2. Pass the event to the view under the mouse.
        # NOTE: Currently, only mouse events have a position. Maybe other events should
        # have them too?
        if isinstance(event, MouseEvent):
            # Find the view under the mouse, if any.
            current_view = self._containing_view(event.pos)

            # If that view is different from the last view...
            # TODO: Add a test for this once multiple views are better supported
            if self._last_mouse_view != current_view:
                # ...send a MouseLeaveEvent to the last view...
                if self._last_mouse_view is not None:
                    self._last_mouse_view.filter_event(MouseLeaveEvent())
                # ...and a MouseEnterEvent to the new view.
                if current_view is not None:
                    current_view.filter_event(
                        MouseEnterEvent(pos=event.pos, buttons=event.buttons)
                    )
            self._last_mouse_view = current_view

            if current_view is not None:
                # 2a. Give the view under the mouse the chance to handle the event.
                if current_view.filter_event(event):
                    return True
                # 2b. If the view didn't handle the event, give any camera controller
                #    on the view the chance to handle it.
                if current_view.camera.interactive:
                    if ctrl := current_view.camera.controller:
                        return ctrl.handle_event(event, current_view)

        # 3. MouseLeave events won't be on any view (because they have no position),
        #    so we need to handle them at the canvas level to clear the last_mouse_view.
        elif isinstance(event, MouseLeaveEvent):
            if self._last_mouse_view is not None:
                self._last_mouse_view.filter_event(event)
                self._last_mouse_view = None

        return False

    def _containing_view(self, pos: tuple[float, float]) -> View | None:
        for view in self.views:
            if view.content_rect is None:
                continue
            x, y, w, h = view.content_rect
            if x <= pos[0] <= x + w and y <= pos[1] <= y + h:
                return view
        return None
