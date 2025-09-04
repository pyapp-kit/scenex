from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, computed_field

from .image import Image, _passes_through_parallelogram
from .node import AABB  # noqa: TC001

if TYPE_CHECKING:
    from scenex.app.events._events import Ray

RenderMode = Literal["iso", "mip"]


class Volume(Image):
    """A dense 3-dimensional array of intensity values."""

    render_mode: RenderMode = Field(
        default="mip",
        description="The method to use in rendering the volume.",
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
        d, w, h = self.data.shape

        # We can describe each face using a parallelogram using:
        # A point for the Top, Left, and Front faces
        tlf = self.transform.map((0, 0, 0, 1))[:3]
        # Or a point for the Bottom, Right, and Back faces
        brb = self.transform.map((w, h, d, 1))[:3]
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
