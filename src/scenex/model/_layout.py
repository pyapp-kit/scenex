from __future__ import annotations

import logging

from cmap import Color
from pydantic import ConfigDict, Field

from ._base import EventedBase

logger = logging.getLogger(__name__)


class Layout(EventedBase):
    """Rectangular layout model with positioning and styling.

    The Layout model defines the position, size, and visual styling of rectangular
    areas. It uses a box model with margin, border, padding, and content areas,
    similar to CSS.

    Examples
    --------
    Create a layout at position (100, 100) with size 400x300:
        >>> layout = Layout(x=100, y=100, width=400, height=300)

    Create a layout with border and padding:
        >>> layout = Layout(
        ...     width=200,
        ...     height=200,
        ...     border_width=2,
        ...     border_color=Color("white"),
        ...     padding=10,
        ... )

    Notes
    -----
    The layout follows this box model::

            y
            |
            v
        x-> +--------------------------------+  ^
            |            margin              |  |
            |  +--------------------------+  |  |
            |  |         border           |  |  |
            |  |  +--------------------+  |  |  |
            |  |  |      padding       |  |  |  |
            |  |  |  +--------------+  |  |  |   height
            |  |  |  |   content    |  |  |  |  |
            |  |  |  |              |  |  |  |  |
            |  |  |  +--------------+  |  |  |  |
            |  |  +--------------------+  |  |  |
            |  +--------------------------+  |  |
            +--------------------------------+  v

            <------------ width ------------->
    """

    x: float = Field(
        default=0, description="The x-coordinate of the left edge of the layout"
    )
    y: float = Field(
        default=0, description="The y-coordinate of the top edge of the layout"
    )
    width: float = Field(
        default=600, description="The total width (including margin, border, padding)"
    )
    height: float = Field(
        default=600, description="The total height (including margin, border, padding)"
    )
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

    @property
    def position(self) -> tuple[float, float]:
        """Return the x, y position of the layout as a tuple."""
        return self.x, self.y

    @property
    def size(self) -> tuple[float, float]:
        """Return the width, height of the layout as a tuple."""
        return self.width, self.height

    @property
    def content_rect(self) -> tuple[float, float, float, float]:
        """Return the (x, y, width, height) of the content area."""
        offset = self.padding + self.border_width + self.margin
        return (
            self.x + offset,
            self.y + offset,
            self.width - 2 * offset,
            self.height - 2 * offset,
        )

    def __contains__(self, pos: tuple[float, float]) -> bool:
        offset = self.padding + self.border_width + self.margin

        left = self.x + offset
        right = self.x + self.width - offset
        bottom = self.y + offset
        top = self.y + self.height - offset
        return left <= pos[0] and pos[0] <= right and bottom <= pos[1] and pos[1] <= top
