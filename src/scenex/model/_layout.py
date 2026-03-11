from __future__ import annotations

import logging
import re
from typing import Any

from cmap import Color
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from ._base import EventedBase

logger = logging.getLogger(__name__)


class Coord(BaseModel):
    """Distance along a number of pixels. Expressed using CSS-style strings."""

    pct: float = Field(
        default=0.0,
        ge=-100,
        le=100,
        description=(
            "Percentage of the total number of pixels (negative values measured back "
            "from the total)"
        ),
    )
    px: int = Field(
        default=0,
        description="Number of pixels (negative values measured back from the total)",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> Any:
        if isinstance(v, str):
            return cls._parse(v)
        if isinstance(v, dict):
            return v
        raise ValueError(f"Invalid Coord value {v!r}")

    @staticmethod
    def _parse(value: str) -> dict:
        v = value.replace(" ", "")
        if not v:
            raise ValueError("Empty Coord string")
        tokens = [t for t in re.split(r"(?=[+-])", v) if t]
        pct, px = 0.0, 0
        for tok in tokens:
            if tok.endswith("%"):
                pct += float(tok[:-1])
            elif tok.endswith("px"):
                px += int(tok[:-2])
            else:
                raise ValueError(f"Invalid Coord term {tok!r}")
        return {"pct": pct, "px": px}

    def resolve(self, total: int) -> int:
        result = int(self.pct / 100 * total) + self.px
        return total + result if result < 0 else result

    @model_serializer
    def _serialize(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        # If comparing to a string, ensure it would parse to the equivalent Coord
        if isinstance(other, str):
            try:
                other = Coord(**Coord._parse(other))
            except (ValueError, Exception):
                return False
        return super().__eq__(other)

    def __str__(self) -> str:
        parts = []
        if self.pct:
            parts.append(f"{self.pct:g}%")
        if self.px or not parts:
            parts.append(f"{self.px}px")
        if len(parts) < 2:
            return parts[0]
        pct_str, px_str = parts
        sep = " " if px_str.startswith("-") else " + "
        return f"{pct_str}{sep}{px_str}"


class Layout(EventedBase):
    """Style model for a view's border, padding, background, and placement.

    Placement is defined by four independent strings — one for each
    edge of the view rect — resolved against the canvas size at render time via
    ``Canvas.rect_for``. Accepted strings must follow CSS conventions:

    * ``"XX%"`` — a percentage of the canvas size along that axis
    * ``"XXpx"`` — a fixed pixel offset; negative values are measured from the
      far edge (right / bottom)

    Examples
    --------
    Full canvas (default)::

        Layout()

    Fixed 400x300 region starting at (50, 50)::

        Layout(x_start="50px", x_end="450px", y_start="50px", y_end="350px")

    Left half, full height::

        Layout(x_end="50%")

    Notes
    -----
    The layout follows this box model::

                  x_start                          x_end
                  |                                |
                  v                                v
        y_start-> +--------------------------------+
                  |            margin              |
                  |  +--------------------------+  |
                  |  |         border           |  |
                  |  |  +--------------------+  |  |
                  |  |  |      padding       |  |  |
                  |  |  |  +--------------+  |  |  |
                  |  |  |  |   content    |  |  |  |
                  |  |  |  |              |  |  |  |
                  |  |  |  +--------------+  |  |  |
                  |  |  +--------------------+  |  |
                  |  +--------------------------+  |
          y_end-> +--------------------------------+
    """

    x_start: Coord = Coord(pct=0)
    x_end: Coord = Coord(pct=100)
    y_start: Coord = Coord(pct=0)
    y_end: Coord = Coord(pct=100)

    @property
    def x(self) -> tuple[Coord, Coord]:
        """The x-axis start/end as a tuple."""
        return self.x_start, self.x_end

    @x.setter
    def x(self, value: tuple[Coord, Coord]) -> None:
        """Set the x-axis start/end from a tuple."""
        self.x_start, self.x_end = value

    @property
    def y(self) -> tuple[Coord, Coord]:
        """The y-axis start/end as a tuple."""
        return self.y_start, self.y_end

    @y.setter
    def y(self, value: tuple[Coord, Coord]) -> None:
        """Set the y-axis start/end from a tuple."""
        self.y_start, self.y_end = value

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
