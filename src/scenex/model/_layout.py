from __future__ import annotations

import logging
import re
from typing import Annotated

from cmap import Color
from pydantic import AfterValidator, ConfigDict, Field

from ._base import EventedBase

logger = logging.getLogger(__name__)


def resolve_dim(value: str, total: int) -> int:
    """Resolve a CSS-style dimension string to an integer pixel position.

    Parameters
    ----------
    value :
        A validated dimension string: ``"XX%"`` (fraction of *total*),
        ``"XXpx"`` (pixel offset; negative values are measured from the far
        edge), or an arithmetic expression such as ``"50% - 20px"`` or
        ``"-40px+100%"``.  Spaces around operators are optional.
    total :
        The canvas size along the relevant axis, in pixels.
    """
    # Remove all whitespace
    v = value.replace(" ", "")
    if not len(v):
        raise ValueError("Empty dimension string")
    # Tokenize into signed terms (e.g. "-40px+100%" → ["-40px", "+100%"])
    tokens = [t for t in re.split(r"(?=[+-])", v) if t]
    # Sum up each token
    result = 0
    for tok in tokens:
        if tok.endswith("%"):
            result += int(float(tok[:-1]) / 100 * total)
        elif tok.endswith("px"):
            result += int(tok[:-2])
        else:
            raise ValueError(f"Invalid dimension term {tok!r}")
    # Negative result is measured from the far edge (e.g. -40px → total-40).
    return total + result if result < 0 else result


def _validate_coord(value: str) -> str:
    """Validate a Unit."""
    # Check to see that it resolves with an arbitrarily large total
    resolve_dim(value, 100000)
    # Then return the original string (with whitespace stripped)
    return value


# A CSS-like length unit
Unit = Annotated[str, AfterValidator(_validate_coord)]


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

    x_start: Unit = "0%"
    x_end: Unit = "100%"
    y_start: Unit = "0%"
    y_end: Unit = "100%"

    @property
    def x(self) -> tuple[Unit, Unit]:
        """The x-axis start/end as a tuple."""
        return self.x_start, self.x_end

    @x.setter
    def x(self, value: tuple[Unit, Unit]) -> None:
        """Set the x-axis start/end from a tuple."""
        self.x_start, self.x_end = value

    @property
    def y(self) -> tuple[Unit, Unit]:
        """The y-axis start/end as a tuple."""
        return self.y_start, self.y_end

    @y.setter
    def y(self, value: tuple[Unit, Unit]) -> None:
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
