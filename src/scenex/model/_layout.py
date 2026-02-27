from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Literal, Union

from cmap import Color
from pydantic import ConfigDict, Field, model_validator

from ._base import EventedBase

if TYPE_CHECKING:
    from scenex.model._canvas import Canvas
    from scenex.model._view import View

logger = logging.getLogger(__name__)

AnyRegion = Annotated[
    Union["PixelRegion", "FractionalRegion", "TiledRegion"], Field(discriminator="type")
]


class Region(EventedBase):
    """Defines how a view is placed on a canvas."""

    @abstractmethod
    def compute_rect(
        self,
        view: View,
        canvas: Canvas,
    ) -> tuple[int, int, int, int]:
        """Return pixel rect (x, y, width, height) for this view given its canvas."""
        ...


class TiledRegion(Region):
    """Just put the view on the canvas.

    This is "reasonable default" region. It's behavior ensures:
    1. If there is only one default view on the canvas, it will fill the entire canvas.
    2. If there are multiple default views on the canvas, they will be stacked
        horizontally in the order they were added, each taking an equal share of the
        canvas width and the full height.
    """

    type: Literal["tiled"] = Field(default="tiled", repr=False)

    def compute_rect(
        self,
        view: View,
        canvas: Canvas,
    ) -> tuple[int, int, int, int]:
        default_views = [
            v for v in canvas.views if isinstance(v.layout.region, TiledRegion)
        ]
        idx = default_views.index(view)
        cw, ch = canvas.size
        n = len(default_views)
        return (int(idx * (cw / n)), 0, int(cw / n), ch)


class FractionalRegion(Region):
    """Positions a view to cover a proportion of its canvas."""

    start: int | tuple[int, int] = Field(default=0)
    end: int | tuple[int, int] = Field(default=1)
    total: int | tuple[int, int] = Field(default=1)

    type: Literal["fractional"] = Field(default="fractional", repr=False)

    def compute_rect(
        self,
        view: View,
        canvas: Canvas,
    ) -> tuple[int, int, int, int]:
        canvas_width, canvas_height = canvas.size
        start_width, start_height = (
            self.start if isinstance(self.start, tuple) else (self.start, self.start)
        )
        end_width, end_height = (
            self.end if isinstance(self.end, tuple) else (self.end, self.end)
        )
        total_width, total_height = (
            self.total if isinstance(self.total, tuple) else (self.total, self.total)
        )
        return (
            int((start_width / total_width) * canvas_width),
            int((start_height / total_height) * canvas_height),
            int(((end_width - start_width) / total_width) * canvas_width),
            int(((end_height - start_height) / total_height) * canvas_height),
        )


class PixelRegion(Region):
    """Positions a view at absolute pixel coordinates within its canvas.

    ``left`` and ``top`` are measured from the **canvas top-left corner**.
    Negative values count from the opposite edge (e.g. ``left=-100`` places
    the left edge 100 px from the canvas right).

    ``right`` and ``bottom`` are also measured from the top-left corner for
    positive values, but ``0`` and negative values count from the **opposite
    canvas edge** (e.g. ``right=0`` is flush with the canvas right edge,
    ``bottom=-40`` is 40 px from the canvas bottom).

    Specify exactly two of ``{left, right, width}`` for the horizontal axis,
    and exactly two of ``{top, bottom, height}`` for the vertical axis.

    Examples
    --------
    40px-wide strip anchored to the left edge, with 40px top/bottom margins:
        >>> region = PixelRegion(left=0, width=40, top=40, bottom=-40)

    Full-width bar anchored to the canvas bottom with a fixed 40px height:
        >>> region = PixelRegion(left=0, right=0, bottom=0, height=40)

    Fill the area right of and below a 40px margin (flush to right/bottom):
        >>> region = PixelRegion(left=40, right=0, top=40, bottom=0)

    100x50 px tile pinned to the top-right corner:
        >>> region = PixelRegion(left=-100, width=100, top=0, height=50)
    """

    type: Literal["pixel"] = Field(default="pixel", repr=False)

    left: int | None = Field(
        default=None,
        description=(
            "X coordinate of the view's left edge from the canvas left. "
            "Negative counts from the canvas right edge."
        ),
    )
    right: int | None = Field(
        default=None,
        description=("X coordinate of the view's right edge from the canvas left. "),
    )
    width: int | None = Field(default=None, description="Width of the view in pixels.")
    top: int | None = Field(
        default=None,
        description=("Y coordinate of the view's top edge from the canvas top. "),
    )
    bottom: int | None = Field(
        default=None,
        description=("Y coordinate of the view's bottom edge from the canvas top. "),
    )
    height: int | None = Field(
        default=None, description="Height of the view in pixels."
    )

    @model_validator(mode="after")
    def _check_constraints(self) -> PixelRegion:
        h_given = sum(v is not None for v in (self.left, self.right, self.width))
        v_given = sum(v is not None for v in (self.top, self.bottom, self.height))
        if h_given < 2:
            raise ValueError(
                "PixelRegion requires at least two of {left, right, width}"
            )
        if v_given < 2:
            raise ValueError(
                "PixelRegion requires at least two of {top, bottom, height}"
            )
        return self

    def compute_rect(
        self,
        view: View,
        canvas: Canvas,
    ) -> tuple[int, int, int, int]:
        cw, ch = canvas.size

        def _start(val: int, size: int) -> int:
            # left/top: negative counts from the far side
            return size + val if val < 0 else val

        def _end(val: int, size: int) -> int:
            # right/bottom: negative count from the far side
            return size + val if val <= 0 else val

        # Resolve horizontal axis
        if self.left is not None and self.right is not None:
            x = _start(self.left, cw)
            w = _end(self.right, cw) - x
        elif self.left is not None and self.width is not None:
            x = _start(self.left, cw)
            w = self.width
        else:  # right + width
            assert self.right is not None and self.width is not None
            x2 = _end(self.right, cw)
            w = self.width
            x = x2 - w

        # Resolve vertical axis
        if self.top is not None and self.bottom is not None:
            y = _start(self.top, ch)
            h = _end(self.bottom, ch) - y
        elif self.top is not None and self.height is not None:
            y = _start(self.top, ch)
            h = self.height
        else:  # bottom + height
            assert self.bottom is not None and self.height is not None
            y2 = _end(self.bottom, ch)
            h = self.height
            y = y2 - h

        return (x, y, w, h)


class Layout(EventedBase):
    """Style model for a view's border, padding, background, and layout region.

    The Layout holds visual styling properties and the region that determines
    how the view is positioned and sized within its canvas. Actual pixel geometry
    (x, y, width, height) is computed on demand by the canvas via
    ``Canvas.rect_for(view)`` and ``Canvas.content_rect_for(view)``.

    Examples
    --------
    Create a layout with a border:
        >>> layout = Layout(border_width=2, border_color=Color("white"), padding=10)

    Use a proportional region (left half of canvas):
        >>> layout = Layout(region=FractionalRegion(end=1, total=2))

    Use a pixel region (40px left strip, 40px top/bottom margin):
        >>> layout = Layout(region=PixelRegion(left=0, width=40, top=40, bottom=40))
    """

    region: AnyRegion = Field(default_factory=TiledRegion)
    background_color: Color | None = Field(
        default=Color("black"),
        description="The background color (inside of the border). "
        "None implies transparent",
    )
    border_width: float = Field(
        default=0, description="The width of the border in pixels."
    )
    border_color: Color | None = Field(
        default=Color("black"), description="The color of the border."
    )
    padding: int = Field(
        default=0,
        description="Number of pixels between border and content",
    )
    margin: int = Field(
        default=0,
        description="Number of pixels between top/left edge and border",
    )

    model_config = ConfigDict(extra="forbid")
