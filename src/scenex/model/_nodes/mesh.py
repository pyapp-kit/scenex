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
        # TODO
        return None
