from __future__ import annotations

import fractions as _fractions
import logging
from abc import abstractmethod
from typing import Literal

from cmap import Color
from pydantic import BaseModel, ConfigDict, Field

from ._base import EventedBase

logger = logging.getLogger(__name__)


class Dim(BaseModel):
    """Abstract base for layout dimensions.

    Each concrete subclass owns its own resolution logic.  Arithmetic on any
    two ``Dim`` values produces a :class:`ComposedDim` that delegates to each
    operand at resolve time.

    Create values with :func:`px` and :func:`fr`::

        px(-40)  # 40px from the far edge
        fr(0.5) - px(200)  # midpoint minus 200px
    """

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def resolve(self, total: int) -> int:
        """Return the integer pixel position for a canvas of the given size."""
        ...

    @abstractmethod
    def __repr__(self) -> str: ...

    def __add__(self, other: AnyDim) -> ComposedDim:
        return ComposedDim(dim1=self, dim2=other, operand="add")  # pyright: ignore[reportArgumentType]

    def __sub__(self, other: AnyDim) -> ComposedDim:
        return ComposedDim(dim1=self, dim2=other, operand="sub")  # pyright: ignore[reportArgumentType]


class ComposedDim(Dim):
    """Result of arithmetic on two :class:`Dim` values."""

    dim1: AnyDim
    dim2: AnyDim
    operand: Literal["add", "sub"]

    def resolve(self, total: int) -> int:
        if self.operand == "add":
            return self.dim1.resolve(total) + self.dim2.resolve(total)
        return self.dim1.resolve(total) - self.dim2.resolve(total)

    def __repr__(self) -> str:
        op = "+" if self.operand == "add" else "-"
        return f"{self.dim1!r} {op} {self.dim2!r}"


class Pixel(Dim):
    """Pixel-only dimension returned by :func:`px`.

    Positive values are measured from the near edge (left / top).
    Negative values are measured from the far edge (right / bottom).
    """

    pixels: int

    def resolve(self, total: int) -> int:
        if self.pixels < 0:
            return total + self.pixels
        return self.pixels

    def __neg__(self) -> Pixel:
        return Pixel(pixels=-self.pixels)

    def __repr__(self) -> str:
        return f"px({self.pixels})"


class Fraction(Dim):
    """Fractional dimension returned by :func:`fr`.

    Stored as an exact rational (``num / denom``) to avoid floating-point
    drift when the same fraction is reused at different canvas sizes.
    """

    num: int
    denom: int

    def resolve(self, total: int) -> int:
        return int(self.num / self.denom * total)

    def __neg__(self) -> Fraction:
        return Fraction(num=-self.num, denom=self.denom)

    def __mul__(self, scalar: float) -> Fraction:
        r = _fractions.Fraction(self.num, self.denom) * _fractions.Fraction(
            scalar
        ).limit_denominator(1000)
        return Fraction(num=r.numerator, denom=r.denominator)

    def __rmul__(self, scalar: float) -> Fraction:
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        val = self.num / self.denom
        frac_str = str(self.num) if self.denom == 1 else repr(val)
        return f"fr({frac_str})"


AnyDim = ComposedDim | Pixel | Fraction

# Resolve the forward reference in ComposedDim.dim1 / dim2.
ComposedDim.model_rebuild()


def px(n: int) -> Pixel:
    """A pixel-only dimension.

    Positive values are measured from the near edge (left / top).
    Negative values are measured from the far edge (right / bottom)::

        px(40)  # 40px from the near edge
        px(-40)  # 40px from the far edge  (equivalent to fr(1) - px(40))
    """
    return Pixel(pixels=n)


def fr(f: float) -> Fraction:
    """A fractional dimension — a proportion of the canvas size.

    ``fr(1)`` is the full canvas width/height; ``fr(0.5)`` is the midpoint.
    Combine with :func:`px` to add a fixed pixel offset::

        fr(0.5) - px(200)  # midpoint minus 200px
    """
    r = _fractions.Fraction(f).limit_denominator(1000)
    return Fraction(num=r.numerator, denom=r.denominator)


class Layout(EventedBase):
    """Style model for a view's border, padding, background, and placement.

    Placement is defined by four independent :class:`Dim` values — one for
    each edge of the view rect — resolved against the canvas size at render
    time via :meth:`~scenex.Canvas.rect_for`.

    Each ``Dim`` is either a :class:`Pixel`, a :class:`Fraction`, or a
    :class:`ComposedDim` combining the two.  Use the :func:`px` and :func:`fr`
    helpers to build them, and combine with ``+`` and ``-``:

    Examples
    --------
    Full canvas (default)::

        Layout()

    Fixed 400x300 region starting at (50, 50)::

        Layout(x_start=px(50), x_end=px(450), y_start=px(50), y_end=px(350))

    Left half, full height::

        Layout(x_end=fr(0.5))

    40px inset on every side::

        Layout(x_start=px(40), x_end=px(-40), y_start=px(40), y_end=px(-40))

    Centered 400px-wide strip::

        Layout(x_start=fr(0.5) - px(200), x_end=fr(0.5) + px(200))

    40px left column, main area fills remainder::

        sidebar = Layout(x_start=px(0), x_end=px(40))
        main = Layout(x_start=px(40), x_end=fr(1))

    Create a layout with a border::

        Layout(border_width=2, border_color=Color("white"), padding=10)
    """

    x_start: AnyDim = Field(default_factory=lambda: fr(0))
    x_end: AnyDim = Field(default_factory=lambda: fr(1))
    y_start: AnyDim = Field(default_factory=lambda: fr(0))
    y_end: AnyDim = Field(default_factory=lambda: fr(1))

    @property
    def x(self) -> tuple[AnyDim, AnyDim]:
        """The x-axis start/end as a tuple."""
        return self.x_start, self.x_end

    @x.setter
    def x(self, value: tuple[AnyDim, AnyDim]) -> None:
        """Set the x-axis start/end from a tuple."""
        self.x_start, self.x_end = value

    @property
    def y(self) -> tuple[AnyDim, AnyDim]:
        """The y-axis start/end as a tuple."""
        return self.y_start, self.y_end

    @y.setter
    def y(self, value: tuple[AnyDim, AnyDim]) -> None:
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
