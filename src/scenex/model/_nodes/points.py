from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from annotated_types import Interval
from cmap import Color
from pydantic import Field

from .node import Node

if TYPE_CHECKING:
    from scenex.events.events import Ray

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
        default=10.0, description="The diameter of the points."
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

    def passes_through(self, ray: Ray) -> float | None:
        # Math graciously adapted from:
        # https://raytracing.github.io/books/RayTracingInOneWeekend.html#addingasphere/ray-sphereintersection

        # Step 1 - Determine whether the ray passes through any points

        # Convert coords to a 3-dimensional numpy array
        coords = np.asarray(self.coords)
        if coords.ndim < len(ray.origin):
            coords = np.pad(
                self.coords, ((0, 0), (0, 1)), mode="constant", constant_values=0
            )
        # And then transform the points to world space
        coords = self.transform.map(coords)[:, :3]

        # For each point, determine whether the ray passes through its sphere.
        #
        # The sphere is defined by the center n=(t, u, v) and a radius r such that any
        # point p=(x, y, z) on the plane satisfies (t-x)^2 + (u-y)^2 + (v-z)^2 = r^2.
        # Note that r is defined in our model as:
        r = self.size / 2 + (self.edge_width if self.edge_width else 0)
        # Note that our intersection point p could be any point along our ray, defined
        # as (ray.origin + ray.direction * t). Substituting this definition into the
        # sphere equation, yields a quadratic equation at^2 + bt + c = 0, where a, b,
        # and c have the following definitions:
        ray_diff = coords - ray.origin
        a = np.dot(ray.direction, ray.direction)
        b = -2 * np.dot(ray_diff, ray.direction)
        c = np.sum(ray_diff * ray_diff, axis=1) - r**2

        # And there is a sphere intersection if the equation's discriminant is
        # non-negative:
        discriminants = b**2 - 4 * a * c
        intersecting_indices = np.where(discriminants >= 0)[0]
        if not intersecting_indices.size:
            return None

        # Step 2 - Determine the depth of intersection

        # We have (potentially) multiple points intersected by our ray, described
        # by the variable t in our ray's definition. Let's focus on those:
        b = b[intersecting_indices]
        discriminants = discriminants[intersecting_indices]
        # We only care about the closest such intersection, i.e. the smallest value of
        # t. Thus, for each intersecting sphere we compute the one or two values of t
        # where our ray intersects. Note that t = (-b +- sqrt(discriminant)) / 2a
        pos_solutions = -b + np.sqrt(discriminants) / (2 * a)
        neg_solutions = -b - np.sqrt(discriminants) / (2 * a)

        # return the smallest such value:
        return float(np.hstack((pos_solutions, neg_solutions)).min())
