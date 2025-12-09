from collections.abc import Sequence
from typing import Literal

from cmap import Color
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class ColorModel(BaseModel):
    """Defines how colors are applied to rendered geometry.

    A ColorModel specifies both the coloring strategy (uniform, per-face, or per-vertex)
    and the color data itself. This allows flexible control over how geometric objects
    are colored in the scene.

    Attributes
    ----------
    type : Literal["uniform", "face", "vertex"]
        The coloring strategy:
        - "uniform": Single color applied to entire geometry
        - "face": One color per face (requires sequence matching face count)
        - "vertex": One color per vertex (requires sequence matching vertex count)
    color : Sequence[Color] | Color
        The color data. Must be:
        - A single Color instance when type="uniform"
        - A sequence of Color instances when type="face" or type="vertex"

    Examples
    --------
    Uniform coloring (single color for entire object):
        >>> ColorModel(type="uniform", color=Color("red"))

    Per-face coloring (each face gets its own color):
        >>> ColorModel(type="face", color=[Color("red"), Color("blue"), Color("green")])

    Per-vertex coloring (colors interpolated across faces):
        >>> ColorModel(
        ...     type="vertex", color=[Color("red"), Color("blue"), Color("green")]
        ... )
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["uniform", "face", "vertex"] = Field(
        default="uniform",
        description="Coloring strategy: 'uniform', 'face', or 'vertex'",
    )
    color: Sequence[Color] | Color = Field(
        union_mode="left_to_right",
        default=Color("white"),
        description="Single Color for uniform, or sequence of Colors for face/vertex",
    )

    @model_validator(mode="after")
    def validate_color_consistency(self) -> Self:
        """Validate that color field matches the specified type."""
        if self.type == "uniform":
            # For uniform coloring, must be a single Color
            if not isinstance(self.color, Color):
                raise ValueError(
                    "For uniform color type, color must be a single Color instance"
                )
        elif self.type in ("face", "vertex"):
            # For face/vertex coloring, must be a sequence of Colors
            if isinstance(self.color, Color):
                raise ValueError(
                    f"For {self.type} color type, color must be a sequence of "
                    "Color instances"
                )
            # Check that it's actually a sequence and all elements are Colors
            try:
                if not isinstance(self.color, Sequence):
                    raise ValueError(
                        f"For {self.type} color type, color must be a sequence of "
                        "Color instances"
                    )
                if not all(isinstance(c, Color) for c in self.color):
                    raise ValueError(
                        f"For {self.type} color type, all elements must be "
                        "Color instances"
                    )
            except (TypeError, AttributeError) as exc:
                raise ValueError(
                    f"For {self.type} color type, color must be a sequence of "
                    "Color instances"
                ) from exc

        return self
