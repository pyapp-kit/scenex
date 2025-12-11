from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from annotated_types import Interval
from cmap import Color
from pydantic import Field

from scenex.model._color import UniformColor, VertexColors

from .node import AABB, Node

if TYPE_CHECKING:
    from scenex.app.events import Ray
    from scenex.model._view import View

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
    """A collection of point markers rendered at specified coordinates.

    Points displays symbols (markers) at 2D or 3D coordinates in the scene. Each point
    is rendered using a specified symbol shape (disc, square, star, etc.) with
    customizable size, face color, and edge styling. Points support different scaling
    modes to control whether they maintain constant screen size or scale with the scene.

    Attributes
    ----------
    coords : array-like
        Array of point coordinates. Shape should be (N, 2) for 2D points or (N, 3) for
        3D points, where N is the number of points.
    size : float
        Diameter of each point marker in pixels. Must be in range [0.5, 500].
    face_color : UniformColor | VertexColors
        Color of the point symbol's interior. Default is white.
    edge_color : UniformColor | VertexColors
        Color of the point symbol's border/outline. Default is black.
    edge_width : float | None
        Width of the point symbol's border in pixels. Default is 1.0.
    symbol : str
        Symbol shape to use for rendering. Available symbols: "disc", "arrow", "ring",
        "clobber", "square", "x", "diamond", "vbar", "hbar", "cross", "tailed_arrow",
        "triangle_up", "triangle_down", "star", "cross_lines".
    scaling : bool | Literal["fixed", "scene", "visual"]
        How points scale when zooming. True/"scene" scales with scene, False/"fixed"
        maintains constant screen size, "visual" scales with visual space.
    antialias : float
        Anti-aliasing amount in pixels for smoother symbol rendering.

    Examples
    --------
    Create simple point markers:
        >>> coords = np.random.rand(100, 2) * 100
        >>> points = Points(coords=coords, size=5, face_color=Color("red"))

    Create points with custom symbols and styling:
        >>> points = Points(
        ...     coords=my_coords,
        ...     symbol="star",
        ...     size=20,
        ...     face_color=Color("yellow"),
        ...     edge_color=Color("orange"),
        ...     edge_width=2,
        ... )

    Create fixed-size points that don't scale with zoom:
        >>> points = Points(
        ...     coords=coords, size=10, scaling=False, face_color=Color("blue")
        ... )

    Create 3D points:
        >>> coords_3d = np.random.rand(50, 3) * 100
        >>> points = Points(coords=coords_3d, symbol="diamond", size=15)
    """

    node_type: Literal["points"] = "points"

    # numpy array of 2D/3D point centers, shape (N, 2) or (N, 3)
    coords: Any = Field(
        default=None,
        repr=False,
        exclude=True,
        description="Array of point coordinates with shape (N, 2) or (N, 3)",
    )
    size: Annotated[float, Interval(ge=0.5, le=500)] = Field(
        default=10.0, description="Diameter of each point marker in pixels"
    )
    face_color: UniformColor | VertexColors = Field(
        default_factory=lambda: UniformColor(color=Color("white")),
        description="Color of the point symbol's interior",
    )
    edge_color: UniformColor | VertexColors = Field(
        default_factory=lambda: UniformColor(color=Color("black")),
        description="Color of the point symbol's border",
    )
    edge_width: float = Field(
        default=1.0, description="Width of the point symbol's border in pixels"
    )
    symbol: SymbolName = Field(
        default="disc",
        description="Symbol shape for rendering (disc, square, star, etc.)",
    )
    # TODO: these are vispy-specific names.  Determine more general names
    scaling: ScalingMode = Field(
        default=True,
        description="Scaling mode: True/scene scales, False/fixed is constant",
    )

    antialias: float = Field(
        default=1, description="Anti-aliasing amount in pixels for smoother rendering"
    )

    @property  # TODO: Cache?
    def bounding_box(self) -> AABB:
        arr = np.asarray(self.coords)
        return (
            tuple(float(d) for d in np.min(arr, axis=0)),
            tuple(float(d) for d in np.max(arr, axis=0)),
        )  # type: ignore

    def passes_through(self, ray: Ray) -> float | None:
        if self.scaling in (False, "fixed"):
            # Note that fixed-size points are tested in screen/canvas space
            # There's then a question of what the returned "distance" means here.
            # For our purposes, consider a plane, perpendicular to the ray, passing
            # through the closest intersected point. The returned distance is then the
            # distance along the ray to that plane.
            return self._passes_through_screen(ray)
        elif self.scaling in (True, "scene"):
            return self._passes_through_world(ray)
        else:  # "scene"
            raise NotImplementedError(
                "Points with 'scene' scaling mode do not (yet) support "
                "ray intersection tests."
            )

    def _passes_through_screen(self, ray: Ray) -> float | None:
        """Test ray intersection in screen/canvas space for fixed-size points."""
        if self.coords is None or len(self.coords) == 0:
            return None

        # Convert points to canvas space
        canvas_points = self._node_to_canvas(ray.source)
        # Convert ray origin to canvas space
        canvas_ray = self._world_to_canvas(ray, np.array([ray.origin]))[0]

        # Calculate distance from ray point to each canvas point
        distances = np.sqrt(
            (canvas_points[:, 0] - canvas_ray[0]) ** 2
            + (canvas_points[:, 1] - canvas_ray[1]) ** 2
        )

        # Determine effective radius in canvas space (size + edge_width)
        # FIXME: If we the model offers an edge mode ala pygfx we will need to adjust
        # this calculation accordingly.
        canvas_radius = self.size / 2 + (self.edge_width / 2 if self.edge_width else 0)

        # Find points that the ray passes through (within radius)
        intersecting_indices = np.where(distances <= canvas_radius)[0]
        if not intersecting_indices.size:
            return None

        # For intersecting points, calculate the world-space distance along the ray
        coords = np.asarray(self.coords)
        if coords.ndim < len(ray.origin):
            coords = np.pad(
                coords,
                ((0, 0), (0, len(ray.origin) - coords.shape[1])),
                mode="constant",
                constant_values=0,
            )

        # Transform intersecting points to world space
        world_coords = self.transform.map(coords[intersecting_indices])[:, :3]

        # Calculate distances along the ray to intersecting points
        ray_to_points = world_coords - ray.origin
        ray_dir_squared = np.dot(ray.direction, ray.direction)

        if ray_dir_squared == 0:
            return None  # Degenerate ray

        # Project onto ray direction to get distances
        distances_along_ray = np.dot(ray_to_points, ray.direction) / ray_dir_squared

        # Only consider intersections in front of the ray (d >= 0)
        valid_distances = distances_along_ray[distances_along_ray >= 0]
        if len(valid_distances) == 0:
            return None

        return float(np.min(valid_distances))

    @staticmethod
    def _world_to_canvas(ray: Ray, points: np.ndarray) -> np.ndarray:
        """Convert world coordinates to canvas coordinates."""
        cam = ray.source.camera
        layout = ray.source.layout
        ndc_points = cam.projection.map(cam.transform.imap(points))[:, :2]
        return (ndc_points + 1) / 2 * (layout.width, layout.height)

    def _node_to_canvas(self, view: View) -> np.ndarray:
        """Convert node coordinates to canvas coordinates."""
        cam = view.camera
        layout = view.layout
        tform_to_root_scene = self.transform_to_node(view.scene)
        ndc_points = cam.projection.map(
            cam.transform.imap(tform_to_root_scene.map(self.coords))
        )[:, :2]
        return np.asarray((ndc_points + 1) / 2 * (layout.width, layout.height))

    def _passes_through_world(self, ray: Ray) -> float | None:
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
        pos_solutions = (-b + np.sqrt(discriminants)) / (2 * a)
        neg_solutions = (-b - np.sqrt(discriminants)) / (2 * a)

        # return the smallest such value:
        return float(np.hstack((pos_solutions, neg_solutions)).min())
