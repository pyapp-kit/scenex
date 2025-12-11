from collections.abc import Sequence
from typing import Any

from cmap import Color
from pydantic import BaseModel, ConfigDict, Field


class ColorModel(BaseModel):
    """
    Base class for color models used in scene nodes.

    This class should not be instantiated directly. Instead, use one of its subclasses:
    - UniformColor: for a single color applied to the entire geometry
    - FaceColors: for per-face coloring (one color per face)
    - VertexColors: for per-vertex coloring (one color per vertex)

    The `color` field is typed as `Any` to allow flexibility in subclasses.
    """

    model_config = ConfigDict(frozen=True)
    color: Any

    def __init__(self, **data: Any) -> None:
        if type(self) is ColorModel:
            raise TypeError("ColorModel cannot be instantiated directly")
        super().__init__(**data)


class UniformColor(ColorModel):
    """
    Uniform coloring strategy for scene nodes.

    This model applies a single color to the entire geometry (mesh, line, points, etc).
    The `color` field is a single `Color` instance (e.g. Color("red")).

    Examples
    --------
    Uniform coloring:
        >>> from cmap import Color
        >>> from scenex import UniformColor
        >>> UniformColor(color=Color("red"))

    """

    color: Color = Field(default_factory=lambda: Color("white"))


class FaceColors(ColorModel):
    """
    Per-face coloring strategy for mesh-like nodes.

    This model applies a different color to each face of a mesh.
    The `color` field is a sequence of `Color` instances, one for each face.

    Examples
    --------
    Per-face coloring:
        >>> from cmap import Color
        >>> from scenex import FaceColors
        >>> FaceColors(color=[Color("red"), Color("blue"), Color("green")])

    """

    color: Sequence[Color]


class VertexColors(ColorModel):
    """
    Per-vertex coloring strategy for mesh, line, or points nodes.

    This model applies a different color to each vertex.
    The `color` field is a sequence of `Color` instances, one for each vertex.

    Examples
    --------
    Per-vertex coloring:
        >>> from cmap import Color
        >>> from scenex import VertexColors
        >>> VertexColors(color=[Color("yellow"), Color("purple"), Color("cyan")])
    """

    color: Sequence[Color]
