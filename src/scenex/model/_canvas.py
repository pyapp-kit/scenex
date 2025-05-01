from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from cmap import Color
from pydantic import ConfigDict, Field

from ._base import EventedBase
from ._evented_list import EventedList
from ._view import View  # noqa: TC001

if TYPE_CHECKING:
    import numpy as np

    from scenex.adaptors._base import CanvasAdaptor


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

    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook for the model."""
        for view in self.views:
            view._canvas = self

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
        adaptor = cast("CanvasAdaptor", self._get_adaptor())
        return adaptor._snx_render()
