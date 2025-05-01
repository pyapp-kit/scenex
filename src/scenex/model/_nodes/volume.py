from typing import Literal

from pydantic import Field

from .image import Image

RenderMode = Literal["iso", "mip"]


class Volume(Image):
    """A dense 3-dimensional array of intensity values."""

    render_mode: RenderMode = Field(
        default="mip",
        description="The method to use in rendering the volume.",
    )
