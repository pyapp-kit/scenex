from typing import Annotated, Any, Literal

from annotated_types import Interval
from cmap import Colormap
from pydantic import Field, computed_field

from .node import AABB, Node

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

    @computed_field  # type: ignore[prop-decorator]
    @property  # TODO: Cache?
    def bounding_box(self) -> AABB:
        if not hasattr(self.data, "shape"):
            raise TypeError(f"{self.data} does not have a shape!")
        shape = self.data.shape
        mi = [-0.5 for _d in shape] + [0] * (3 - len(shape))
        ma = [d - 0.5 for d in shape] + [0] * (3 - len(shape))
        return (tuple(mi), tuple(ma))  # type: ignore
