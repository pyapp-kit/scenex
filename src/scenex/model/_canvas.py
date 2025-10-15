from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from cmap import Color
from pydantic import ConfigDict, Field, PrivateAttr

from scenex.adaptors import get_adaptor_registry
from scenex.app import app
from scenex.app.events import Ray, ResizeEvent

from ._base import EventedBase
from ._evented_list import EventedList
from ._view import View  # noqa: TC001

if TYPE_CHECKING:
    import numpy as np

    from scenex.adaptors._base import CanvasAdaptor
    from scenex.app.events import Event, EventFilter


class Canvas(EventedBase):
    """Canvas onto which views are rendered.

    In desktop applications, this will be a window. In web applications, this will be a
    div.  The canvas has one or more views, which are rendered onto it.  For example,
    an orthoviewer might be a single canvas with three views, one for each axis.
    """

    width: int = Field(default=500, description="The width of the canvas in pixels.")
    height: int = Field(default=500, description="The height of the canvas in pixels.")
    background_color: Color = Field(
        default=Color("black"), description="The background color."
    )
    visible: bool = Field(default=False, description="Whether the canvas is visible.")
    title: str = Field(default="", description="The title of the canvas.")
    views: EventedList[View] = Field(default_factory=EventedList, frozen=True)
    _event_filter: EventFilter | None = PrivateAttr()

    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook for the model."""
        for view in self.views:
            view._canvas = self

        native = get_adaptor_registry().get_adaptor(self)._snx_get_native()
        self._event_filter = app().install_event_filter(native, self)

    @property
    def size(self) -> tuple[int, int]:
        """Return the size of the canvas."""
        return self.width, self.height

    @size.setter
    def size(self, value: tuple[int, int]) -> None:
        """Set the size of the canvas."""
        self.width, self.height = value

    def handle(self, event: Event) -> bool:
        # TODO: Implement the rest in a later PR
        if isinstance(event, ResizeEvent):
            # TODO: How might some event filter tap into the resize?
            self.size = (event.width, event.height)
        return False

    def to_world(self, canvas_pos: tuple[float, float]) -> Ray | None:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""
        # TODO: Implement in a later PR
        return Ray(
            origin=(0, 0, 0),
            direction=(0.0, 0.0, -1.0),
        )

    def render(self) -> np.ndarray:
        """Show the canvas."""
        if adaptors := self._get_adaptors():
            return cast("CanvasAdaptor", adaptors[0])._snx_render()
        raise RuntimeError("No adaptor found for Canvas.")
