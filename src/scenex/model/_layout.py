from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Annotated, Literal, Union

from cmap import Color
from pydantic import ConfigDict, Field

from ._base import EventedBase

logger = logging.getLogger(__name__)

AnySpan = Annotated[
    Union["OffsetPlusSize", "Fractional", "PixelGaps"], Field(discriminator="type")
]


class Span(EventedBase):
    """Defines how a view is placed on a canvas."""

    @abstractmethod
    def resolve(
        self,
        total: int,
    ) -> tuple[int, int]:
        """Return pixel span [start, stop) given a canvas."""
        ...


class Fractional(Span):
    """Defines a span as a constant proportion."""

    start: int = Field(default=0)
    end: int = Field(default=1)
    total: int = Field(default=1)

    type: Literal["fractional"] = Field(default="fractional", repr=False)

    def resolve(
        self,
        total: int,
    ) -> tuple[int, int]:
        resolved_start = int((self.start / self.total) * total)
        resolved_end = int((self.end / self.total) * total)
        return (resolved_start, resolved_end - resolved_start)


class OffsetPlusSize(Span):
    """Defines a span by its start offset and width.

    TODO: Work pixel into the class name
    """

    type: Literal["start_offset"] = Field(default="start_offset", repr=False)

    offset: int = Field(
        description="Offset of the span. Negative counts from the canvas right edge.",
    )
    size: int = Field(
        description="Length of the span.",
    )

    def resolve(
        self,
        total: int,
    ) -> tuple[int, int]:
        start = self.offset if self.offset >= 0 else total + self.offset
        return (start, self.size)


class PixelGaps(Span):
    """Defines a span by the number of pixels to leave as a gap on each side.

    TODO: Better names than left and right
    """

    type: Literal["pixel_gaps"] = Field(default="pixel_gaps", repr=False)

    left: int = Field(
        default=0,
        description=(
            "Number of pixels to leave as a gap on the left. "
            "Negative counts from the right edge."
        ),
    )
    right: int = Field(
        default=0,
        description=(
            "Number of pixels to leave as a gap on the right. "
            "Negative counts from the left edge."
        ),
    )

    def resolve(
        self,
        total: int,
    ) -> tuple[int, int]:
        start = self.left if self.left >= 0 else total + self.left
        end = total - self.right if self.right >= 0 else total + self.right
        return (start, end - self.left)


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
        >>> layout = Layout(x_span=Fractional(start=0, end=1, total=2))

    Use a pixel region (40px left strip, 40px top/bottom margin):
        >>> layout = Layout(
        ...     x_span=OffsetPlusSize(offset=0, size=40),
        ...     y_span=PixelGaps(left=40, right=40),
        ... )
    """

    x_span: AnySpan = Field(default_factory=lambda: Fractional(start=0, end=1, total=1))
    y_span: AnySpan = Field(default_factory=lambda: Fractional(start=0, end=1, total=1))
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
