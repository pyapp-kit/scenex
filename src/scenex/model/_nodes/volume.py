from typing import Literal

from pydantic import Field, computed_field

from scenex.model._nodes.node import AABB

from .image import Image

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
