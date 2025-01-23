from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from cmap import Color
from pydantic import Field

from scenex.model._base import _AT

from ._base import EventedBase, SupportsVisibility
from ._evented_list import EventedList
from .view import View  # noqa: TC001

if TYPE_CHECKING:
    import numpy as np


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

    # show and render will trigger a backend connection

    def show(self) -> None:
        """Show the canvas."""
        self.visible = True

    def hide(self) -> None:
        """Hide the canvas."""
        self.visible = False


# -------------------- Controller ABC --------------------

_CT = TypeVar("_CT", bound="Canvas", covariant=True)


class CanvasAdaptor(SupportsVisibility[_CT, _AT]):
    """Protocol defining the interface for a Canvas adaptor."""

    @abstractmethod
    def _snx_set_width(self, arg: int) -> None: ...
    @abstractmethod
    def _snx_set_height(self, arg: int) -> None: ...
    @abstractmethod
    def _snx_set_background_color(self, arg: Color | None) -> None: ...
    @abstractmethod
    def _snx_set_title(self, arg: str) -> None: ...
    @abstractmethod
    def _vis_close(self) -> None: ...
    @abstractmethod
    def _vis_render(self) -> np.ndarray: ...
    @abstractmethod
    def _vis_add_view(self, view: View) -> None: ...

    def _snx_set_views(self, views: list[View]) -> None:
        pass

    def _snx_get_ipython_mimebundle(
        self, *args: Any, **kwargs: Any
    ) -> dict | tuple[dict, dict] | Any:
        return NotImplemented
