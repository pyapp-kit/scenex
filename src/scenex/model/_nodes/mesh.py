from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from cmap import Color
from pydantic import Field, computed_field

from scenex.model._color import FaceColors, UniformColor, VertexColors

from .node import AABB, Node

if TYPE_CHECKING:
    from scenex.app.events._events import Ray


class Mesh(Node):
    """A 3D surface mesh composed of triangular faces.

    Mesh represents a 3D surface defined by vertices and triangular faces. Each face is
    specified by three indices into the vertex array, forming a triangle. The mesh uses
    counter-clockwise winding order: for a face (v1, v2, v3), the normal vector points
    in the direction of (v2 - v1) x (v3 - v1).

    Meshes support ray-triangle intersection testing using the Möller-Trumbore
    algorithm, enabling efficient picking and interaction.

    Attributes
    ----------
    vertices : array-like
        Array of vertex positions. Shape should be (N, 2) or (N, 3) where N is the
        number of vertices. For 2D vertices, z-coordinate is assumed to be 0.
    faces : array-like
        Array of face definitions. Shape should be (M, 3) where M is the number of
        triangular faces. Each row contains three indices into the vertices array,
        defining a triangle with counter-clockwise winding.
    color : UniformColor | FaceColors | VertexColors
        Color specification for the mesh. Can be:
        - UniformColor: Single color for the entire mesh
        - FaceColors: One color per face
        - VertexColors: One color per vertex, interpolated across faces

    Examples
    --------
    Create a simple triangle mesh:
        >>> import numpy as np
        >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        >>> faces = np.array([[0, 1, 2]])
        >>> mesh = Mesh(
        ...     vertices=vertices,
        ...     faces=faces,
        ...     color=UniformColor(color=Color("blue")),
        ... )

    Create a square made of two triangles:
        >>> vertices = np.array(
        ...     [
        ...         [0, 0, 0],  # bottom-left
        ...         [1, 0, 0],  # bottom-right
        ...         [1, 1, 0],  # top-right
        ...         [0, 1, 0],  # top-left
        ...     ]
        ... )
        >>> faces = np.array(
        ...     [
        ...         [0, 1, 2],  # first triangle
        ...         [0, 2, 3],  # second triangle
        ...     ]
        ... )
        >>> mesh = Mesh(vertices=vertices, faces=faces)

    Notes
    -----
    Face winding order (counter-clockwise) determines which side of the triangle is
    considered the "front" face. The normal vector for face (v1, v2, v3) points in
    the direction of (v2 - v1) x (v3 - v1).
    """

    node_type: Literal["mesh"] = "mesh"

    # numpy array of 2D/3D vertices, shape (N, 2) or (N, 3)
    vertices: Any = Field(
        default=None,
        repr=False,
        exclude=True,
        description="Array of vertex positions with shape (N, 2) or (N, 3)",
    )
    # Note that the normal vector of each face (v1, v2, v3) is given by
    # n = (v2 - v1) x (v3 - v1).
    faces: Any = Field(
        default=None,
        repr=False,
        exclude=True,
        description="Array of face indices with shape (M, 3) defining triangles",
    )

    color: UniformColor | FaceColors | VertexColors = Field(
        default_factory=lambda: UniformColor(color=Color("white")),
        description="Color specification; uniform, per-face, or per-vertex",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property  # TODO: Cache?
    def bounding_box(self) -> AABB:
        arr = np.asarray(self.vertices)
        return (
            tuple(float(d) for d in np.min(arr, axis=0)),
            tuple(float(d) for d in np.max(arr, axis=0)),
        )  # type: ignore

    def intersecting_faces(self, ray: Ray) -> list[tuple[int, float]]:
        """
        Find all faces that intersect with the given ray.

        Uses the Möller-Trumbore intersection algorithm, vectorized over all triangles.
        Adapted from https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm#C++_implementation

        Parameters
        ----------
        ray : Ray
            The ray to test for intersections.

        Returns
        -------
        list[tuple[int, float]]
            A list of tuples containing (face_index, distance) for each
            intersecting face. Sorted by distance (closest first).
        """
        tracked_faces = np.arange(len(self.faces))

        # Suppose the triangle is defined by vertices v1, v2, v3
        # Barycentric coordinates are given by
        # P = (1 - u - v)*v1 + u*v2 + v*v3,
        #   = v1 + u(v2 - v1) + v(v3 - v1)
        #   = v1 + u*e1 + v*e2
        # But the intersection point is also given by our ray equation:
        # P = ray.origin + t*ray.direction
        # So we need to solve the equation
        # ray.origin + t*ray.direction = v1 + u*e1 + v*e2
        # Rearranging:
        # ray.origin - v1 = -t*ray.direction + u*e1 + v*e2
        e1 = self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]]
        e2 = self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]]

        # First, cull all triangles parallel to the ray
        # We compute the determinant (scalar triple product) for this
        ray_cross_e2 = np.cross(ray.direction, e2)
        # NOTE: Vectorized version of row-wise dot product of ray_cross_e2 and e1
        det = np.sum(ray_cross_e2 * e1, axis=1)
        parallel_triangles = np.isclose(det, 0)

        # Remove parallel triangles from consideration
        e1 = e1[~parallel_triangles]
        e2 = e2[~parallel_triangles]
        ray_cross_e2 = ray_cross_e2[~parallel_triangles]
        det = det[~parallel_triangles]
        v1 = self.vertices[self.faces[:, 0]][~parallel_triangles]
        tracked_faces = tracked_faces[~parallel_triangles]

        # We can use Cramer's Rule to solve for t, u, v
        # (We solve for u, v first to check if the intersection is within the triangle)
        #
        # u = (1/det) * scalar_triple_product(ray.direction, s, e2)
        inv_det = 1 / det
        s = ray.origin - v1
        u = inv_det * np.sum(s * ray_cross_e2, axis=1)
        # v = (1/det) * scalar_triple_product(ray.direction, e1, s)
        s_cross_e1 = np.cross(s, e1)
        v = inv_det * np.sum(ray.direction * s_cross_e1, axis=1)

        # Cull triangles where the intersection is outside the triangle
        intersecting = (u >= 0) & (v >= 0) & (u + v < 1)

        if not np.any(intersecting):
            return []

        # Get the indices and data for intersecting triangles
        tracked_faces = tracked_faces[intersecting]
        inv_det = inv_det[intersecting]
        e2 = e2[intersecting]
        s_cross_e1 = s_cross_e1[intersecting]

        # t = (1/det) * scalar_triple_product(s, e1, e2)
        t = inv_det * np.sum(e2 * s_cross_e1, axis=1)

        # Create list of (face_index, distance) tuples and sort by distance
        intersections = list(zip(tracked_faces, t, strict=True))
        intersections.sort(key=lambda x: x[1])  # Sort by distance

        return intersections

    def passes_through(self, ray: Ray) -> float | None:
        """
        Check if the ray passes through this mesh and return the closest distance.

        Parameters
        ----------
        ray : Ray
            The ray to test for intersection.

        Returns
        -------
        float | None
            The distance to the closest intersection, or None if no intersection.
        """
        intersections = self.intersecting_faces(ray)
        if not intersections:
            return None

        # Return the closest intersection distance
        return float(intersections[0][1])
