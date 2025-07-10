from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from annotated_types import Interval
from cmap import Colormap
from pydantic import Field

from .node import Node

if TYPE_CHECKING:
    from scenex.events.events import Ray

InterpolationMode = Literal["nearest", "linear", "bicubic"]


class Image(Node):
    """A dense array of intensity values."""

    node_type: Literal["image"] = Field(default="image", repr=False)

    # NB: we may want this to be a pure `set_data()` method, rather than a field
    # on the model that stores state.
    data: Any = Field(
        default=None, repr=False, exclude=True, description="The current image data."
    )
    cmap: Colormap = Field(
        default_factory=lambda: Colormap("gray"),
        description="The colormap to apply when rendering the image.",
    )
    clims: tuple[float, float] | None = Field(
        default=None,
        description="The min and max values to use when normalizing the image.",
    )
    gamma: Annotated[float, Interval(gt=0, le=2)] = Field(
        default=1.0, description="Gamma correction applied after normalization."
    )
    interpolation: InterpolationMode = Field(
        default="nearest", description="Interpolation mode."
    )

    def passes_through(self, ray: Ray) -> float | None:
        origin = self.transform.map((0, 0, 0, 1))[:3]
        u = self.transform.map((self.data.shape[0], 0, 0, 0))[:3]
        v = self.transform.map((0, self.data.shape[1], 0, 0))[:3]
        return _passes_through_parallelogram(ray, origin, u, v)


def _passes_through_parallelogram(
    ray: Ray, origin: np.ndarray, u: np.ndarray, v: np.ndarray
) -> float | None:
    """Determine whether a ray passes through a parallelogram defined by (origin, u, v).

    Parameters
    ----------
    ray : Ray
        The ray passing through the scene
    origin : np.ndarray
        A np.ndarray of shape (3,) representing the origin point of the parallelogram.
    u : np.ndarray
        A np.ndarray of shape (3,) representing the direction and length of one edge of
        the parallelogram.
    v : np.ndarray
        A np.ndarray of shape (3,) representing the direction and length of another edge
        of the parallelogram. Note that u and v should not be parallel.

    Returns
    -------
    t: float | None
        The depth t at which the ray intersects the node, or None if it never
        intersects.
    """
    # Math graciously adapted from:
    # https://raytracing.github.io/books/RayTracingTheNextWeek.html#quadrilaterals

    # Step 1 - Determine where the ray intersects the image plane

    # The image plane is defined by the normal vector n=(a, b, c) and an offset (d)
    # such that any point p=(x, y, z) on the plane satisfies np.dot(v, p) = d, or
    # ax + by + cz + -d = 0.

    # In this case, the normal vector n can be found by the cross product of u and v
    tformed = np.cross(u, v)
    normal = tformed / np.linalg.norm(tformed)
    # And we know that the origin of the image is on the plane. Using that point we can
    # find d...
    d = np.dot(normal, origin)
    # ... and with d we can find the depth t at which the ray would intersect the plane.
    #
    # Note that our ray is defined by (ray.origin + ray.direction * t).
    # This is just np.dot(normal, ray.origin + ray.direction * t) = d,
    # rearranged to solve for t.
    ray_normal_inner_product = np.dot(normal, ray.direction)
    if ray_normal_inner_product == 0:
        # Plane is parallel to the ray, so no intersection.
        return None
    t = (d - np.dot(normal, ray.origin)) / ray_normal_inner_product
    # With our value of t, we can find the intersection point:
    intersection = tuple(
        a + t * b for a, b in zip(ray.origin, ray.direction, strict=False)
    )

    # Step 2 - Determine whether the ray hits the image.

    # We need to determine whether the planar intersection is within the image
    # interval bounds. In other words, the intersection point should be within
    # [0, magnitude(u)] units away from the image origin along the u axis and
    # [0, magnitude(v)] units away from the image origin along the v axis.
    offset = intersection - origin

    # We use some fancy math derived from the link above to convert offset into...
    n = np.cross(u, v)
    w = n / np.dot(n, n)
    # ...the component of offset in direction of u...
    alpha = np.dot(w, np.cross(offset, v))
    # ...and the component of offset in direction of v
    beta = np.dot(w, np.cross(u, offset))

    # Our ray passes through the image if alpha and beta are within [0, 1]
    is_inside = alpha >= 0 and alpha <= 1 and beta >= 0 and beta <= 1

    # If the ray passes through node, return the depth of the intersection.
    return t if is_inside else None
