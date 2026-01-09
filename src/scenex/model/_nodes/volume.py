from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, computed_field

from .image import Image, _passes_through_parallelogram
from .node import AABB  # noqa: TC001

if TYPE_CHECKING:
    from scenex.app.events._events import Ray

RenderMode = Literal["iso", "mip"]


class Volume(Image):
    """A 3D volumetric dataset rendered with volume rendering techniques.

    Volume extends Image to support 3D volumetric data. Unlike images which are 2D
    arrays, volumes are 3D arrays of intensity values that are rendered using volume
    rendering techniques like maximum intensity projection (MIP) or isosurface
    rendering.

    The volume uses ZYX dimension ordering, meaning data.shape = (depth, height, width).
    Like Image, the volume supports colormapping, intensity normalization, and gamma
    correction. The rendering mode determines how the 3D data is projected onto the 2D
    viewing plane.

    Attributes
    ----------
    render_mode : Literal["iso", "mip"]
        Volume rendering method:
        - "mip": Maximum Intensity Projection - shows the maximum value along each ray
        - "iso": Isosurface rendering - renders a surface at a specific intensity value

    Examples
    --------
    Create a volume with MIP rendering:
        >>> import numpy as np
        >>> data = np.random.rand(50, 100, 100)  # ZYX dimensions
        >>> volume = Volume(data=data, render_mode="mip")

    Create a volume with custom colormap and intensity range:
        >>> volume = Volume(
        ...     data=data,
        ...     cmap=Colormap("viridis"),
        ...     clims=(0, 1),
        ...     render_mode="iso",
        ... )

    Notes
    -----
    Volume inherits all Image attributes including data, cmap, clims, gamma, and
    interpolation. The data should be a 3D array with shape (depth, height, width)
    following ZYX convention.
    """

    render_mode: RenderMode = Field(
        default="mip", description="Volume rendering method"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property  # TODO: Cache?
    def bounding_box(self) -> AABB:
        bb = super().bounding_box
        # We can reuse the image version, but the first dimension needs to be swapped
        # To account for the ZYX convention.
        return ((bb[0][1], bb[0][2], bb[0][0]), (bb[1][1], bb[1][2], bb[1][0]))

    def passes_through(self, ray: Ray) -> float | None:
        # The ray passes through our volume if it passes through any of the six faces
        mi, ma = self.bounding_box
        d, w, h = self.data.shape

        # We can describe each face using a parallelogram using:
        # A point for the Top, Left, and Front faces
        tlf = self.transform.map((mi[0], mi[1], mi[2], 1))[:3]
        # Or a point for the Bottom, Right, and Back faces
        brb = self.transform.map((ma[0], ma[1], ma[2], 1))[:3]
        # As well as vectors describing the three edges eminating from tlf
        u = self.transform.map((w, 0, 0, 0))[:3]
        v = self.transform.map((0, h, 0, 0))[:3]
        w = self.transform.map((0, 0, d, 0))[:3]

        faces = [
            # (origin, edge1, edge2)
            (tlf, u, v),  # front face
            (tlf, v, w),  # left face
            (tlf, w, u),  # top face
            (brb, -u, -v),  # back face
            (brb, -v, -w),  # right face
            (brb, -w, -u),  # bottom face
        ]
        # Compute the depths where the ray intersects each face
        results = [_passes_through_parallelogram(ray, o, e1, e2) for o, e1, e2 in faces]
        # And return the minimum depth in the case of multiple intersections.
        depths = [r for r in results if r is not None]
        return min(depths) if depths else None
