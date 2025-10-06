from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from cmap import Color
from pydantic import Field, computed_field

from .node import AABB, Node

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from scenex.app.events._events import Ray


class Line(Node):
    """A line or polyline defined by a sequence of vertices.

    Lines are drawn by connecting consecutive vertices in the order they appear.
    """

    node_type: Literal["line"] = "line"

    # numpy array of 2D/3D vertices, shape (N, 2) or (N, 3)
    vertices: Any = Field(default=None, repr=False, exclude=True)

    color: Color | None = Field(
        default=Color("white"), description="The color of the line."
    )
    width: float = Field(default=1.0, ge=0.0, description="The width of the line.")
    antialias: float = Field(default=1.0, description="Anti-aliasing factor, in px.")

    @computed_field  # type: ignore[prop-decorator]
    @property  # TODO: Cache?
    def bounding_box(self) -> AABB:
        arr = np.asarray(self.vertices)
        return (
            tuple(float(d) for d in np.min(arr, axis=0)),
            tuple(float(d) for d in np.max(arr, axis=0)),
        )  # type: ignore

    def passes_through(self, ray: Ray) -> float | None:
        """
        Check if the ray passes through this line and return the closest distance.

        For lines, this checks intersection with line segments formed by
        consecutive vertices.

        Parameters
        ----------
        ray : Ray
            The ray to test for intersection.

        Returns
        -------
        float | None
            The distance to the closest intersection, or None if no intersection.
        """
        vertices = np.asarray(self.vertices)
        if len(vertices) < 2:
            return None

        min_distance = float("inf")
        found_intersection = False

        # Check each line segment
        for i in range(len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]

            # Calculate intersection with line segment
            distance = self._ray_line_segment_intersection(ray, v1, v2)
            if distance is not None and distance < min_distance:
                min_distance = distance
                found_intersection = True

        return float(min_distance) if found_intersection else None

    def _ray_line_segment_intersection(
        self, ray: Ray, v1: np.ndarray, v2: np.ndarray
    ) -> float | None:
        """
        Calculate intersection distance between a ray and a line segment.

        Uses the closest point approach for 3D line-line intersection.
        """
        if ray.source is None:
            return None

        def to_canvas(point: ArrayLike) -> tuple[float, float]:
            """Convert a 3D point to 2D canvas coordinates."""
            cam = ray.source.camera
            ndc = cam.projection.map(cam.transform.imap(point))
            layout = ray.source.layout
            return (
                (ndc[0] + 1) / 2 * layout.width,
                (ndc[1] + 1) / 2 * layout.height,
            )

        v1_canvas = to_canvas(v1)
        v2_canvas = to_canvas(v2)
        ray_canvas = to_canvas(ray.origin)

        num = np.abs(
            (v2_canvas[1] - v1_canvas[1]) * ray_canvas[0]
            - (v2_canvas[0] - v1_canvas[0]) * ray_canvas[1]
            + v2_canvas[0] * v1_canvas[1]
            - v2_canvas[1] * v1_canvas[0]
        )
        den = np.sqrt(
            (v2_canvas[1] - v1_canvas[1]) ** 2 + (v2_canvas[0] - v1_canvas[0]) ** 2
        )

        distance = num / den if den != 0 else float("inf")
        print(distance)
        if distance > self.width:
            return None
        # Calculate distance along the ray direction
        a = np.subtract(ray_canvas, v1_canvas)
        b = np.subtract(v2_canvas, v1_canvas)
        t = np.dot(a, b) / np.dot(b, b)
        if t < 0 or t > 1:
            return None
        intersect_world = v1 + t * (v2 - v1)

        # Calculate the distance along the ray to the intersection point
        # The ray is defined as: ray.origin + d * ray.direction
        # We need to find d such that ray.origin + d * ray.direction = intersect_world
        # This gives us: d * ray.direction = intersect_world - ray.origin
        # Solving for d: d = dot(intersect_world - ray.origin, ray.direction) /
        #                     dot(ray.direction, ray.direction)
        ray_to_intersect = np.subtract(intersect_world, ray.origin)
        ray_dir_squared = np.dot(ray.direction, ray.direction)

        if ray_dir_squared == 0:
            return None  # Degenerate ray

        d = np.dot(ray_to_intersect, ray.direction) / ray_dir_squared

        # Only return positive distances (intersections in front of the ray)
        return float(d) if d >= 0 else None
