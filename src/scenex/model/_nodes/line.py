from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from cmap import Color
from pydantic import Field, computed_field

from .node import AABB, Node

if TYPE_CHECKING:
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

        min_vals = tuple(float(d) for d in np.min(arr, axis=0))
        max_vals = tuple(float(d) for d in np.max(arr, axis=0))

        # Ensure we have at least 3 dimensions by padding with zeros if needed
        if len(min_vals) == 2:
            min_vals = (*min_vals, 0.0)
            max_vals = (*max_vals, 0.0)

        return (min_vals, max_vals)  # type: ignore

    def passes_through(self, ray: Ray) -> float | None:
        """
        Check if the ray passes through this line.

        Parameters
        ----------
        ray : Ray
            The ray to test for intersection.

        Returns
        -------
        float | None
            The distance to the closest intersection, or None if no intersection.
        """
        verts = np.asarray(self.vertices)
        # Convert vertices to canvas space
        canvas_vertices = Line._world_to_canvas(ray, self.vertices)
        canvas_ray = Line._world_to_canvas(ray, np.array([ray.origin]))[0]

        starts = canvas_vertices[:-1]
        ends = canvas_vertices[1:]

        # Compute the distance from the ray ON THE CANVAS to the closest point the line
        # associated with each line segment.
        #
        # Equation loaned from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
        num = np.abs(
            (ends[:, 1] - starts[:, 1]) * canvas_ray[0]
            - (ends[:, 0] - starts[:, 0]) * canvas_ray[1]
            + ends[:, 0] * starts[:, 1]
            - ends[:, 1] * starts[:, 0]
        )
        den = np.sqrt(
            (ends[:, 1] - starts[:, 1]) ** 2 + (ends[:, 0] - starts[:, 0]) ** 2
        )
        den[den == 0] = float("inf")  # Avoid division by zero
        distance = num / den

        # Determine the corresponding point in world space corresponding to that closest
        # point. Note that this point is only on the line segment if 0 <= t <= 1.
        # (We check this at the end.)
        a = np.subtract(canvas_ray, starts)
        b = np.subtract(ends, starts)
        # Vectorized version of dot product
        t = np.sum(a * b, axis=1) / np.sum(b * b, axis=1)
        intersect_world = verts[1:] + t[:, np.newaxis] * (verts[:-1] - verts[1:])

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

        # Our ray intersects the line if:
        # 1. The distance from the ray to the line is less than the line width
        # 2. The intersection point is within the line segment (0 <= t <= 1)
        # 3. The intersection point is in front of the ray origin (d >= 0)
        condition = (distance <= self.width) & (t >= 0) & (t <= 1) & (d >= 0)
        valid_intersections = d[condition]
        if len(valid_intersections):
            return float(np.min(valid_intersections))
        else:
            return None

    @staticmethod
    def _world_to_canvas(ray: Ray, points: np.ndarray) -> np.ndarray:
        cam = ray.source.camera
        layout = ray.source.layout
        ndc_points = cam.projection.map(cam.transform.imap(points))[:, :2]
        return (ndc_points + 1) / 2 * (layout.width, layout.height)
