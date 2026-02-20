from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from cmap import Colormap
from pydantic import Field

from .node import AABB, Node

if TYPE_CHECKING:
    from scenex.app.events._events import Ray

InterpolationMode = Literal["nearest", "linear", "bicubic"]


class Image(Node):
    """A 2D image rendered as a textured rectangle.

    Image displays a 2D array of intensity values, mapping them to colors using a
    colormap. The image is rendered as a rectangle in 3D space, with pixels centered
    at integer coordinates starting from (0, 0). The image supports various rendering
    options including colormapping, intensity normalization, gamma correction, and
    interpolation.

    The image's geometry spans from (-0.5, -0.5) to (width-0.5, height-0.5), meaning
    that pixel centers are at integer coordinates. This convention aligns with standard
    image processing practices.

    Examples
    --------
    Create a simple grayscale image:
        >>> import numpy as np
        >>> data = np.random.rand(100, 100)
        >>> img = Image(data=data)

    Create an image with custom colormap and intensity range:
        >>> img = Image(data=data, cmap=Colormap("viridis"), clims=(0, 255))

    Create a transformed and semi-transparent image:
        >>> img = Image(
        ...     data=data,
        ...     transform=Transform().translated((10, 20)).scaled((2, 2)),
        ...     opacity=0.7,
        ... )

    Apply gamma correction to brighten dark images:
        >>> img = Image(data=data, gamma=0.5)
    """

    node_type: Literal["image"] = Field(default="image", repr=False)

    # NB: we may want this to be a pure `set_data()` method, rather than a field
    # on the model that stores state.
    data: Any = Field(
        default=None,
        repr=False,
        exclude=True,
        description="2D array of intensity values",
    )
    cmap: Colormap = Field(
        default_factory=lambda: Colormap("gray"),
        description="Converts intensity values to colors",
    )
    clims: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Min/max intensity values for normalization. "
            "Values outside this range are clipped. "
            "None uses data range"
        ),
    )
    gamma: float = Field(
        default=1.0,
        gt=0,
        le=2,
        description="Gamma correction factor. Applied after normalization",
    )
    interpolation: InterpolationMode = Field(
        default="nearest",
        description="Defines color interpolation method between data values.",
    )

    @property  # TODO: Cache?
    def bounding_box(self) -> AABB:
        if not hasattr(self.data, "shape"):
            raise TypeError(f"{self.data} does not have a shape!")
        shape = self.data.shape
        mi = [-0.5 for _d in shape] + [0] * (3 - len(shape))
        ma = [d - 0.5 for d in shape] + [0] * (3 - len(shape))
        return (tuple(mi), tuple(ma))  # type: ignore

    def passes_through(self, ray: Ray) -> float | None:
        mi, _ma = self.bounding_box
        origin = self.transform.map(mi)[:3]
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
    # [0, magnitude(u)) units away from the image origin along the u axis and
    # [0, magnitude(v)) units away from the image origin along the v axis.
    #
    # Note the open right bound: if the ray intersects exactly on the far edge of the
    # image, we consider that a miss. This approach serves two important purposes:
    #
    # 1. Adjacent image handling: When images are tiled side-by-side, this prevents
    #    ambiguity at shared boundaries (only one image reports intersection).
    #
    # 2. Array indexing safety: The [0, 1) bounds ensure that subsequent coordinate-to-
    #    array-index mapping never produces out-of-bounds indices. Alternatively we'd
    #    require clamping logic during array access.
    offset = intersection - origin

    # We use some fancy math derived from the link above to convert offset into...
    n = np.cross(u, v)
    w = n / np.dot(n, n)
    # ...the component of offset in direction of u...
    alpha = np.dot(w, np.cross(offset, v))
    # ...and the component of offset in direction of v
    beta = np.dot(w, np.cross(u, offset))

    # Our ray passes through the image if alpha and beta are within [0, 1)
    is_inside = alpha >= 0 and alpha < 1 and beta >= 0 and beta < 1

    # If the ray passes through node, return the depth of the intersection.
    return float(t) if is_inside else None
