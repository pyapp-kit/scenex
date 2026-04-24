from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from cmap import Color
from pydantic import ConfigDict, Field
from typing_extensions import Unpack

from scenex.model._evented_list import EventedList

from ._base import EventedBase
from ._view import View  # noqa: TC001

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import TypedDict

    from scenex.adaptors._base import CanvasAdaptor

    class CanvasKwargs(TypedDict, total=False):
        width: int
        height: int
        background_color: Color
        visible: bool
        title: str


class Canvas(EventedBase):
    """A rendering surface that displays one or more views.

    The Canvas represents the top-level rendering context where views are displayed.
    In desktop applications, a canvas corresponds to a window. In web applications,
    it corresponds to a DOM element. Multiple views can be arranged on a single canvas
    using their layout parameters.

    Canvas is a pure data model. Attach a ``CanvasInteractor`` to enable event
    routing and interaction.

    Examples
    --------
    Create a simple canvas with default settings:
        >>> canvas = Canvas()

    Create a canvas with custom size and title:
        >>> canvas = Canvas(width=800, height=600, title="My Visualization")

    Create a canvas with multiple views side-by-side:
        >>> canvas = Canvas(width=800, height=400, views=[View(), View()])

    Attach interaction::

        from scenex.interaction import CanvasInteractor, PanZoom

        ci = CanvasInteractor(canvas)
        ci.set_controller(view, PanZoom())
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
        frozen=True,
    )

    model_config = ConfigDict(extra="forbid")

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            views: Sequence[View] = (),
            **data: Unpack[CanvasKwargs],
        ) -> None: ...

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook for the model."""
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

    def _on_view_inserted(self, _: int, view: View) -> None:
        if view.canvas is not self:
            view.canvas = self

    def _on_view_removed(self, _: int, view: View) -> None:
        if view.canvas is self:
            view.canvas = None

    def _on_view_changed(
        self,
        _: int | slice,
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
        """Render the canvas to a numpy array."""
        if adaptors := self._get_adaptors():
            return cast("CanvasAdaptor", adaptors[0])._snx_render()
        raise RuntimeError("No adaptor found for Canvas.")
