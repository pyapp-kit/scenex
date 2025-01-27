from typing import Any, Literal

from cmap import Colormap
from pydantic import Field

from .node import Node

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
    gamma: float = Field(
        default=1.0, description="Gamma correction applied after normalization."
    )
    interpolation: InterpolationMode = Field(
        default="nearest", description="Interpolation mode."
    )
