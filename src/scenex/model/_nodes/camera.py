from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PrivateAttr, computed_field

from scenex.model._transform import Transform
from scenex.utils import projections

from .node import Node

if TYPE_CHECKING:
    from collections.abc import Callable

    from scenex.events.events import Event, Ray
    from scenex.model._transform import Transform

CameraType = Literal["panzoom", "perspective"]
Position2D = tuple[float, float]
Position3D = tuple[float, float, float]
Position = Position2D | Position3D


class Camera(Node):
    """A camera that defines the view and perspective of a scene.

    The camera lives in, and is a child of, a scene graph.  It defines the view
    transformation for the scene, mapping it onto a 2D surface.

    Cameras have two different Transforms. Like all Nodes, it has a transform
    `transform`, describing its location in the world. Its other transform,
    `projection`, describes how 2D normalized device coordinates
    {(x, y) | x in [-1, 1], y in [-1, 1]} map to a ray in 3D world space. The inner
    product of these matrices can convert a 2D canvas position to a 3D ray eminating
    from the camera node into the world.
    """

    def __init__(
        self, controller: Callable[[Event, Node], bool] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.set_event_filter(controller)

    node_type: Literal["camera"] = "camera"

    interactive: bool = Field(
        default=True,
        description="Whether the camera responds to user interaction, "
        "such as mouse and keyboard events.",
    )
    projection: Transform = Field(
        default_factory=lambda: projections.orthographic(2, 2, 2),
        description="Describes how 3D points are mapped to a 2D canvas, "
        "default is an orthographic projection of a 2x2x2 cube, centered at (0, 0, 0)",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property  # TODO: Cache?
    def bounding_box(self) -> None:
        # Prevent cameras from distorting scene bounding boxes
        return None

    _filter: Callable[[Event, Node], bool] | None = PrivateAttr(default=None)

    def passes_through(self, ray: Ray) -> float | None:
        return None
