from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from cmap import Colormap
from pydantic import Field

from .node import Node, NodeController

if TYPE_CHECKING:
    from numpy.typing import NDArray


InterpolationMode = Literal["nearest", "bilinear", "bicubic"]


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


# -------------------- Controller ABC --------------------

_IT = TypeVar("_IT", bound="Image", covariant=True)


class ImageController(NodeController[_IT]):
    """Protocol for a backend Image adaptor object."""

    @abstractmethod
    def _vis_set_data(self, arg: NDArray) -> None: ...
    @abstractmethod
    def _vis_set_cmap(self, arg: Colormap) -> None: ...
    @abstractmethod
    def _vis_set_clims(self, arg: tuple[float, float] | None) -> None: ...
    @abstractmethod
    def _vis_set_gamma(self, arg: float) -> None: ...
    @abstractmethod
    def _vis_set_interpolation(self, arg: InterpolationMode) -> None: ...
