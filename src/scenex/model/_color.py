from collections.abc import Sequence
from typing import Literal

from cmap import Color
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class ColorModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["uniform", "face", "vertex"]
    color: Sequence[Color] | Color = Field(union_mode="left_to_right")

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
