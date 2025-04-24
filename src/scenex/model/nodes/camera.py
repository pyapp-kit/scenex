from __future__ import annotations

from typing import Literal, Union

from pydantic import Field

from .node import Node

CameraType = Literal["panzoom", "perspective"]
Position2D = tuple[float, float]
Position3D = tuple[float, float, float]
Position = Union[Position2D, Position3D]


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
    center: Position = Field(
        default=(0, 0, 0), description="Center position of the view."
    )
