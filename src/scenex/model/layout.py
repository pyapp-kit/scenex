from __future__ import annotations

import logging

from cmap import Color
from pydantic import ConfigDict, Field, computed_field

from ._base import EventedBase

logger = logging.getLogger(__name__)


class Layout(EventedBase):
    """Rectangular layout model.

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
        default=0, description="The x-coordinate of the object (wrt parent)."
    )
    y: float = Field(
        default=0, description="The y-coordinate of the object (wrt parent)."
    )
    width: float = Field(default=600, description="The width of the object.")
    height: float = Field(default=600, description="The height of the object.")
    background_color: Color | None = Field(
        default=Color("black"),
        description="The background color (inside of the border). "
        "None implies transparent.",
    )
    border_width: float = Field(
        default=0, description="The width of the border in pixels."
    )
    border_color: Color | None = Field(
        default=Color("black"), description="The color of the border."
    )
    padding: int = Field(
        default=0,
        description="The amount of padding in the widget "
        "(i.e. the space reserved between the contents and the border).",
    )
    margin: int = Field(
        default=0, description="The margin to keep outside the widget's border"
    )

    model_config = ConfigDict(extra="forbid")

    @computed_field  # type: ignore [prop-decorator]
    @property
    def position(self) -> tuple[float, float]:
        """Return the x, y position of the layout as a tuple."""
        return self.x, self.y

    @computed_field  # type: ignore [prop-decorator]
    @property
    def size(self) -> tuple[float, float]:
        """Return the width, height of the layout as a tuple."""
        return self.width, self.height
