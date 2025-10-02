from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from cmap import Color
from pydantic import Field, computed_field

from .node import AABB, Node

if TYPE_CHECKING:
    from scenex.app.events._events import Ray


class Mesh(Node):
    """A surface of triangular faces.

    Each face is defined by a 3-tuple of indices into a list of nD vertices.
    """

    node_type: Literal["mesh"] = "mesh"

    # numpy array of 2D/3D vertices, shape (N, 2) or (N, 3)
    vertices: Any = Field(default=None, repr=False, exclude=True)
    # Note that the normal vector of each face (v1, v2, v3) is given by
    # n = (v2 - v1) x (v3 - v1).
    faces: Any = Field(default=None, repr=False, exclude=True)

    # TODO: There are many different ways to color a mesh. E.g.
    # - per-face color
    # - per-vertex color
    # - texture mapping
    color: Color | None = Field(
        default=Color("white"), description="The color of the mesh."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property  # TODO: Cache?
    def bounding_box(self) -> AABB:
        arr = np.asarray(self.vertices)
        return (
            tuple(float(d) for d in np.min(arr, axis=0)),
            tuple(float(d) for d in np.max(arr, axis=0)),
        )  # type: ignore

    def passes_through(self, ray: Ray) -> float | None:
        # Möller-Trumbore intersection algorithm, vectorized over all triangles
        # Adapted from https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm#C++_implementation

        # TODO: Better documentation. I don't yet understand the math
        e1 = self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]]
        e2 = self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]]
        # Ignore triangles parallel to the ray
        ray_cross_e2 = np.cross(ray.direction, e2)
        # Vectorized version of row-wise dot product of ray_cross_e2 and e1
        det = np.sum(ray_cross_e2 * e1, axis=1)
        parallel_triangles = np.isclose(det, 0)

        # Refactor variables to avoid parallel triangles
        e1 = e1[~parallel_triangles]
        e2 = e2[~parallel_triangles]
        ray_cross_e2 = ray_cross_e2[~parallel_triangles]
        det = det[~parallel_triangles]
        v1 = self.vertices[self.faces[:, 0]][~parallel_triangles]

        inv_det = 1 / det
        s = ray.origin - v1
        u = inv_det * np.sum(s * ray_cross_e2, axis=1)

        s_cross_e1 = np.cross(s, e1)
        v = inv_det * np.sum(ray.direction * s_cross_e1, axis=1)

        intersecting = (u >= 0) & (v >= 0) & (u + v < 1)

        if not np.any(intersecting):
            return None
        inv_det = inv_det[intersecting]
        e2 = e2[intersecting]
        s_cross_e1 = s_cross_e1[intersecting]

        t = inv_det * np.sum(e2 * s_cross_e1, axis=1)
        print(f"{ray.origin}: {t}")

        return float(np.min(t))
