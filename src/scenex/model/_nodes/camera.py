from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import pylinalg as la
from pydantic import Field, computed_field

from scenex.model._transform import Transform
from scenex.utils import projections

from .node import Node

if TYPE_CHECKING:
    from scenex.app.events._events import Ray
    from scenex.model._controller import InteractionStrategy
    from scenex.model._transform import Transform

CameraType = Literal["panzoom", "perspective"]
Position2D = tuple[float, float]
Position3D = tuple[float, float, float]
Vector3D = tuple[float, float, float]
Position = Position2D | Position3D


class Camera(Node):
    """A camera that defines the viewing perspective and projection for a scene.

    The Camera is a node in the scene graph that determines how 3D world space is
    projected onto a 2D canvas. It combines a view transformation (positioning the
    camera in the scene) with a projection transformation (defining the viewing volume
    and perspective).

    Cameras use two transforms:
    - `transform` (inherited from Node): Maps local 3D space to world 3D space,
      positioning and orienting the camera in the scene.
    - `projection`: Maps normalized device coordinates [-1, 1] x [-1, 1] to rays in
      local 3D space, defining the viewing volume and projection type.

    The camera uses a right-handed coordinate system following OpenGL conventions:
    the positive x-axis points right, the positive y-axis points up, and the positive
    z-axis points out of the screen toward the viewer.

    Attributes
    ----------
    controller : InteractionStrategy | None
        Strategy for handling user interactions (e.g., PanZoom, Orbit). If None, no
        interaction strategy is used.
    interactive : bool
        Whether the camera responds to user input events (mouse, keyboard).
    projection : Transform
        The projection transformation mapping 2D normalized device coordinates to 3D
        rays. Default is an orthographic projection of a 2x2x2 cube centered at origin.

    Examples
    --------
    Create a camera with pan-zoom interaction:
        >>> camera = Camera(controller=PanZoom(), interactive=True)

    Create a camera with orbit interaction:
        >>> camera = Camera(controller=Orbit(center=(0, 0, 0)), interactive=True)

    Position a camera and point it at a target:
        >>> camera = Camera()
        >>> camera.transform = Transform().translated((10, 0, 0))
        >>> camera.look_at((0, 0, 0), up=(0, 0, 1))

    Create a perspective camera:
        >>> from scenex.utils.projections import perspective
        >>> camera = Camera(
        ...     projection=perspective(fov=60, aspect=1.5, near=0.1, far=100)
        ... )
    """

    node_type: Literal["camera"] = "camera"

    controller: InteractionStrategy | None = Field(
        default=None,
        description="Strategy for handling user interactions with the camera",
    )
    interactive: bool = Field(
        default=True,
        description="Whether the camera responds to user interaction events",
    )
    projection: Transform = Field(
        default_factory=lambda: projections.orthographic(2, 2, 2),
        description="Projection transformation mapping NDC to 3D rays in local space",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property  # TODO: Cache?
    def bounding_box(self) -> None:
        # Prevent cameras from distorting scene bounding boxes
        return None

    def passes_through(self, ray: Ray) -> float | None:
        # Cameras are not rendered objects
        return None

    @property
    def forward(self) -> Vector3D:
        """The forward direction of the camera in world space, as a unit vector."""
        position = self.transform.map((0, 0, 0))[:3]
        further = self.transform.map((0, 0, -1))[:3]
        vector = further - position
        return tuple(vector / np.linalg.norm(vector))

    @forward.setter
    def forward(self, arg: Vector3D) -> None:
        """Sets the forward direction of the camera."""
        # Check for no change - avoid divide-by-zeroes
        mag_old = np.linalg.norm(self.forward)
        mag_new = np.linalg.norm(arg)
        if abs(np.dot(self.forward, arg) / (mag_old * mag_new) - 1) < 1e-3:
            return  # No change needed
        # Compute the quaternion needed to rotate from the current forward direction to
        # the desired forward direction
        rot_quat = la.quat_from_vecs(self.forward, arg)
        rot_axis, rot_angle = la.quat_to_axis_angle(rot_quat)

        # Rotate around the camera's current position
        position = self.transform.map((0, 0, 0))[:3]
        self.transform = (
            self.transform.translated(-position)
            .rotated(rot_angle * 180 / math.pi, rot_axis)
            .translated(position)
        )

    @property
    def up(self) -> Vector3D:
        """The up direction of the camera in world space, as a unit vector."""
        return tuple(self.transform.map((0, 1, 0)) - self.transform.map((0, 0, 0)))[:3]

    @up.setter
    def up(self, arg: Vector3D) -> None:
        """Sets the up direction of the camera.

        Does not affect the forward direction of the camera so long as the new up
        direction is perpendicular to the existing forward direction.
        """
        # Check for no change - avoid divide-by-zeroes
        mag_old = np.linalg.norm(self.up)
        mag_new = np.linalg.norm(arg)
        if abs(np.dot(self.up, arg) / (mag_old * mag_new) - 1) < 1e-3:
            return  # No change needed
        # Compute the quaternion needed to rotate from the current up direction to
        # the desired up direction
        rot_quat = la.quat_from_vecs(self.up, arg)
        rot_axis, rot_angle = la.quat_to_axis_angle(rot_quat)

        # Rotate around the camera's current position
        position = self.transform.map((0, 0, 0))[:3]
        self.transform = (
            self.transform.translated(-position)
            .rotated(rot_angle * 180 / math.pi, rot_axis)
            .translated(position)
        )

    def look_at(self, target: Position3D, /, *, up: Vector3D | None = None) -> None:
        """Adjusts the camera to look at a target point in the world.

        Parameters
        ----------
        target: Position3D
            The position in 3D space that the camera should look at.
        up: Vector3D, optional
            The up direction for the camera. If provided, this vector must be
            perpendicular to the forward vector that results from looking at target.
        """
        position = self.transform.map((0, 0, 0))[:3]
        self.forward = tuple(target - position)
        if up is not None:
            if np.linalg.norm(up) == 0:
                raise ValueError("Up vector must be non-zero.")
            if np.abs(np.dot(self.forward, up)) > 1e-6:
                raise ValueError("Up vector must be perpendicular to forward vector.")
            self.up = up
