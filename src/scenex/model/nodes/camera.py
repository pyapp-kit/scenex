from __future__ import annotations

from abc import abstractmethod
from typing import Literal, TypeVar

from pydantic import Field

from scenex.model._base import _AT

from .node import Node, NodeAdaptor

CameraType = Literal["panzoom", "perspective"]


class Camera(Node):
    """A camera that defines the view of a scene."""

    node_type: Literal["camera"] = "camera"

    type: CameraType = Field(default="panzoom", description="Camera type.")
    interactive: bool = Field(
        default=True,
        description="Whether the camera responds to user interaction, "
        "such as mouse and keyboard events.",
    )
    zoom: float = Field(default=1.0, description="Zoom factor of the camera.")
    center: tuple[float, float, float] | tuple[float, float] = Field(
        default=(0, 0, 0), description="Center position of the view."
    )


# -------------------- Controller ABC --------------------

_CT = TypeVar("_CT", bound=Camera, covariant=True)


class CameraAdaptor(NodeAdaptor[_CT, _AT]):
    """Protocol for a backend camera adaptor object."""

    @abstractmethod
    def _snx_set_type(self, arg: CameraType) -> None: ...
    @abstractmethod
    def _snx_set_zoom(self, arg: float) -> None: ...
    @abstractmethod
    def _snx_set_center(self, arg: tuple[float, ...]) -> None: ...
    @abstractmethod
    def _snx_set_range(self, margin: float) -> None: ...
