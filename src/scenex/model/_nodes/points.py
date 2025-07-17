from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
from annotated_types import Interval
from cmap import Color
from pydantic import Field, computed_field

from .node import AABB, Node

SymbolName = Literal[
    "disc",
    "arrow",
    "ring",
    "clobber",
    "square",
    "x",
    "diamond",
    "vbar",
    "hbar",
    "cross",
    "tailed_arrow",
    "triangle_up",
    "triangle_down",
    "star",
    "cross_lines",
]
ScalingMode = Literal[True, False, "fixed", "scene", "visual"]


class Points(Node):
    """Coordinates that can be represented in a scene."""

    node_type: Literal["points"] = "points"

    # numpy array of 2D/3D point centers, shape (N, 2) or (N, 3)
    coords: Any = Field(default=None, repr=False, exclude=True)
    size: Annotated[float, Interval(ge=0.5, le=100)] = Field(
        default=10.0, description="The size of the points."
    )
    face_color: Color | None = Field(
        default=Color("white"), description="The color of the faces."
    )
    edge_color: Color | None = Field(
        default=Color("black"), description="The color of the edges."
    )
    edge_width: float | None = Field(default=1.0, description="The width of the edges.")
    symbol: SymbolName = Field(
        default="disc", description="The symbol to use for the points."
    )
    # TODO: these are vispy-specific names.  Determine more general names
    scaling: ScalingMode = Field(
        default=True, description="Determines how points scale when zooming."
    )

    antialias: float = Field(default=1, description="Anti-aliasing factor, in px.")

    @computed_field  # type: ignore[prop-decorator]
    @property  # TODO: Cache?
    def bounding_box(self) -> AABB:
        arr = np.asarray(self.coords)
        return (
            tuple(float(d) for d in np.min(arr, axis=0)),
            tuple(float(d) for d in np.max(arr, axis=0)),
        )  # type: ignore
