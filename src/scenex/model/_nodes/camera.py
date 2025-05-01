from __future__ import annotations

from typing import Literal

from pydantic import Field

from .node import Node

CameraType = Literal["panzoom", "perspective"]
Position2D = tuple[float, float]
Position3D = tuple[float, float, float]
Position = Position2D | Position3D


class Camera(Node):
    """A camera that defines the view and perspective of a scene.

    The camera lives in, and is a child of, a scene graph.  It defines the view
    transformation for the scene, mapping it onto a 2D surface.
    """

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
