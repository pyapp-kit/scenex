from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

import numpy as np
import pylinalg as la
from pydantic import Field, PrivateAttr

from scenex.app.events import (
    MouseButton,
    MouseEvent,
    MouseMoveEvent,
    MousePressEvent,
    WheelEvent,
)
from scenex.model._base import EventedBase
from scenex.utils import projections

from .node import Node

if TYPE_CHECKING:
    from scenex.app.events import Event
    from scenex.model._transform import Transform

Position2D = tuple[float, float]
Position3D = tuple[float, float, float]
Vector3D = tuple[float, float, float]
Position = Position2D | Position3D

AnyController = Annotated[
    Union["PanZoom", "Orbit", "None"], Field(discriminator="type")
]


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

    Examples
    --------
    Create a camera with pan-zoom controller:
        >>> camera = Camera(controller=PanZoom(), interactive=True)

    Create a camera with orbit controller:
        >>> camera = Camera(controller=Orbit(center=(0, 0, 0)), interactive=True)

    Position a camera and point it at a target:
        >>> camera = Camera()
        >>> camera.transform = Transform().translated((10, 0, 0))
        >>> camera.look_at((0, 0, 0), up=(0, 0, 1))

    Create a perspective camera:
        >>> from scenex.utils.projections import perspective
        >>> camera = Camera(projection=perspective(fov=70, near=0.1, far=100))
    """

    node_type: Literal["camera"] = "camera"

    controller: AnyController = Field(
        default=None,
        description="Describes how user interaction affects the camera",
    )
    interactive: bool = Field(
        default=True,
        description="Whether the camera responds to user interaction events",
    )
    projection: Transform = Field(
        default_factory=lambda: projections.orthographic(2, 2, 2),
        description="Transformation mapping NDC to 3D rays in local space",
    )

    @property  # TODO: Cache?
    def bounding_box(self) -> None:
        # Prevent cameras from distorting scene bounding boxes
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


# ====================================================================================
# Camera Controllers
# ====================================================================================


class CameraController(EventedBase):
    """Base class defining how a camera responds to user interaction events.

    A CameraController handles user input (mouse, keyboard, wheel) to manipulate
    camera transforms and projections, enabling interactive behaviors like panning,
    zooming, orbiting, or custom camera controls. Controllers are attached to Camera
    instances via the `controller` field and automatically receive events when the
    camera is marked as `interactive=True`.

    Event handlers should return True if they fully handled the event (stopping further
    propagation) or False if other handlers should continue processing the event.

    Examples
    --------
    Create a camera with pan/zoom controller:
        >>> camera = Camera(controller=PanZoom(), interactive=True)

    Create a camera with orbit controller:
        >>> camera = Camera(controller=Orbit(center=(0, 0, 0)), interactive=True)

    See Also
    --------
    PanZoom : 2D pan and zoom controller
    Orbit : 3D orbit controller
    Camera : Camera class that uses controllers
    """

    @abstractmethod
    def handle_event(self, event: Event, camera: Camera) -> bool:
        """
        Handle a user interaction event to control the camera.

        This method is called automatically on all events on the camera's view that were
        not handled by previous handlers during scenex event processing.

        Parameters
        ----------
        event : Event
            The input event to handle (MouseMoveEvent, MousePressEvent, WheelEvent,
            KeyPressEvent, etc.)
        camera : Camera
            The camera node to manipulate.

        Returns
        -------
        bool
            True if the event was fully handled and should not propagate to other
            handlers, False if not handled or other handlers should process it.
        """
        raise NotImplementedError


class PanZoom(CameraController):
    """2D pan and zoom controller for orthographic views.

    PanZoom provides intuitive mouse-based navigation for 2D scenes and orthographic
    projections.

    The strategy operates in two complementary ways:
    - **Panning** (left mouse drag): Modifies camera.transform to translate the camera
      position, maintaining the scene coordinates under the cursor.
    - **Zooming** (mouse wheel): Modifies camera.projection to scale the view, then
      adjusts camera.transform to keep the zoom centered on the cursor position.

    Optional axis locking allows constraining interaction to horizontal or vertical
    movement only

    Attributes
    ----------
    lock_x : bool
        If True, prevent horizontal panning and zooming. Movement is constrained to
        the vertical axis only. Default is False.
    lock_y : bool
        If True, prevent vertical panning and zooming. Movement is constrained to
        the horizontal axis only. Default is False.

    Examples
    --------
    Standard 2D pan and zoom:
        >>> camera = Camera(controller=PanZoom(), interactive=True)

    Lock horizontal movement for vertical scrolling only:
        >>> camera = Camera(controller=PanZoom(lock_x=True), interactive=True)

    Create an image viewer with pan/zoom:
        >>> import numpy as np
        >>> from scenex.utils import projections
        >>> my_data = np.random.rand(512, 512).astype(np.float32)
        >>> view = View(
        ...     scene=Scene(children=[Image(data=my_data)]),
        ...     camera=Camera(
        ...         controller=PanZoom(),
        ...         interactive=True,
        ...     ),
        ... )
        >>> projections.zoom_to_fit(view=view, type="orthographic")

    See Also
    --------
    Orbit : 3D orbit controller for perspective views
    CameraController : Base class for camera controllers
    Camera : Camera class with controller field
    """

    lock_x: bool = Field(
        default=False,
        description="If True, prevent horizontal panning and zooming.",
    )
    lock_y: bool = Field(
        default=False,
        description="If True, prevent vertical panning and zooming.",
    )
    type: Literal["pan_zoom"] = Field(default="pan_zoom", repr=False)

    # Private state for tracking interactions
    _drag_pos: tuple[float, float] | None = PrivateAttr(default=None)

    def handle_event(self, event: Event, camera: Camera) -> bool:
        """Handle mouse and wheel events to pan/zoom the camera."""
        from scenex.app.events import (
            MouseButton,
            MouseMoveEvent,
            MousePressEvent,
            WheelEvent,
        )

        if not camera.interactive:
            return False

        handled = False

        # Panning involves keeping a particular position underneath the cursor.
        # That position is recorded on a left mouse button press.
        if isinstance(event, MousePressEvent) and MouseButton.LEFT in event.buttons:
            self._drag_pos = event.world_ray.origin[:2]
        # Every time the cursor is moved, until the left mouse button is released,
        # We translate the camera such that the position is back under the cursor
        # (i.e. under the world ray origin)
        elif (
            isinstance(event, MouseMoveEvent)
            and MouseButton.LEFT in event.buttons
            and self._drag_pos
        ):
            new_pos = event.world_ray.origin[:2]
            dx = self._drag_pos[0] - new_pos[0]
            if not self.lock_x:
                camera.transform = camera.transform.translated((dx, 0))
            dy = self._drag_pos[1] - new_pos[1]
            if not self.lock_y:
                camera.transform = camera.transform.translated((0, dy))
            handled = True

        # Note that while panning adjusts the camera's transform matrix, zooming
        # adjusts the projection matrix.
        elif isinstance(event, WheelEvent):
            # Zoom while keeping the position under the cursor fixed.
            _dx, dy = event.angle_delta
            if dy:
                # Step 1: Adjust the projection matrix to zoom in or out.
                zoom = self._zoom_factor(dy)
                camera.projection = camera.projection.scaled(
                    (1 if self.lock_x else zoom, 1 if self.lock_y else zoom, 1.0)
                )

                # Step 2: Adjust the transform matrix to maintain the position
                # under the cursor. The math is largely borrowed from
                # https://github.com/pygfx/pygfx/blob/520af2d5bb2038ec309ef645e4a60d502f00d181/pygfx/controllers/_panzoom.py#L164

                # Find the distance between the world ray and the camera
                zoom_center = np.asarray(event.world_ray.origin)[:2]
                camera_center = np.asarray(camera.transform.map((0, 0)))[:2]
                # Compute the world distance before the zoom
                delta_screen1 = zoom_center - camera_center
                # Compute the world distance after the zoom
                delta_screen2 = delta_screen1 * zoom
                # The pan is the difference between the two
                pan = (delta_screen2 - delta_screen1) / zoom
                camera.transform = camera.transform.translated(
                    (
                        pan[0] if not self.lock_x else 0,
                        pan[1] if not self.lock_y else 0,
                    )
                )
                handled = True

        return handled

    def _zoom_factor(self, delta: float) -> float:
        # Magnifier stolen from pygfx
        return 2 ** (delta * 0.001)


class Orbit(CameraController):
    """3D orbit controller for rotating around a focal point.

    Orbit provides intuitive 3D navigation for perspective views by allowing the camera
    to rotate around a fixed center point while maintaining its distance.

    The strategy uses spherical coordinates to control camera position:
    - **Azimuth**: Rotation around the polar axis (typically Z), controlling left/right
      movement around the scene
    - **Elevation**: Angle from the polar axis, controlling up/down viewing angle
    - **Distance**: Radius from the center point, controlled by zooming

    During rotation, foreground objects (between the camera and the center) move in the
    direction of mouse movement, providing intuitive control where the visible scene
    appears to rotate under the mouse.

    The right mouse button allows panning the orbit center itself, enabling exploration
    of large scenes by moving the focal point while maintaining the camera's viewing
    angle and distance.

    Attributes
    ----------
    center : tuple[float, float, float]
        The point in 3D space around which the camera orbits. This is the focal point
        that remains stationary during rotation. Default is (0, 0, 0).
    polar_axis : tuple[float, float, float]
        The axis defining the "up" direction for orbit calculations. Azimuth rotations
        occur around this axis. Default is (0, 0, 1) for Z-up orientation.

    Examples
    --------
    Orbit around the origin:
        >>> from scenex.utils import projections
        >>> # Create a perspective camera...
        >>> camera = Camera(
        ...     interactive=True,
        ...     projection=projections.perspective(fov=70, near=1, far=1000),
        ... )
        >>> # ...positioned along the X axis...
        >>> camera.transform = Transform().translated((100, 0, 0))
        >>> # ...looking at the origin...
        >>> camera.look_at((0, 0, 0), up=(0, 0, 1))
        >>> # ...that orbits around the origin
        >>> camera.controller = Orbit(center=(0, 0, 0))

    Orbit around a data volume's center:
        >>> import numpy as np
        >>> my_data = np.random.rand(100, 100, 100).astype(np.float32)
        >>> volume = Volume(data=my_data)
        >>> center = np.mean(volume.bounding_box, axis=0)
        >>> # Create a perspective camera...
        >>> camera = Camera(
        ...     interactive=True,
        ...     projection=projections.perspective(fov=70, near=1, far=1000),
        ... )
        >>> # ...positioned along the X axis from the volume center...
        >>> camera.transform = Transform().translated(center).translated((100, 0, 0))
        >>> # ...looking at the center...
        >>> camera.look_at(center, up=(0, 0, 1))
        >>> # ...that orbits around the center
        >>> camera.controller = Orbit(center=center)

    Custom polar axis for Y-up scenes:
        >>> camera = Camera(
        ...     controller=Orbit(center=(0, 0, 0), polar_axis=(0, 1, 0)),
        ...     interactive=True,
        ... )

    Interactions
    ------------
    - **Left mouse drag**: Orbit/rotate the camera around the center point
    - **Right mouse drag**: Pan the orbit center (translates the focal point)
    - **Mouse wheel**: Zoom toward/away from center (change radius)

    Notes
    -----
    Elevation is automatically clamped to [0°, 180°] to prevent the camera from
    going upside down. Without this clamping, the camera could rotate past the
    polar axis, causing horizontal mouse movement to make the foreground rotate
    in the opposite direction to the actual mouse movement.

    See Also
    --------
    PanZoom : 2D pan and zoom controller for orthographic views
    CameraController : Base class for camera controllers
    Camera : Camera class with controller field
    Camera.look_at : Method to orient camera toward a point
    """

    center: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="The point in 3D space around which the camera orbits.",
    )
    polar_axis: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 1.0),
        description='The axis defining the "up" direction for orbit calculations.',
    )
    type: Literal["orbit"] = Field(default="orbit", repr=False)

    # Private state for tracking interactions
    _last_canvas_pos: tuple[float, float] | None = PrivateAttr(default=None)
    _pan_ray: Any = PrivateAttr(default=None)  # Ray type

    def handle_event(self, event: Event, camera: Camera) -> bool:
        """Handle mouse and wheel events to orbit the camera."""
        if not camera.interactive:
            return False

        handled = False
        center_array = np.asarray(self.center)

        # Orbit on mouse move with left button held
        if (
            isinstance(event, MouseMoveEvent)
            and event.buttons == MouseButton.LEFT
            and self._last_canvas_pos is not None
        ):
            # The process of orbiting is as follows:
            # 1. Compute the azimuth and elevation changes based on mouse movement.
            #   - Azimuth describes the angle between the the positive X axis and
            #       the projection of the camera's position onto the XY plane.
            #   - Elevation describes the angle between the camera's position and
            #       the positive Z axis.
            # 2. Ensure these changes are clamped to valid ranges (only really
            #   applies to elevation).
            # 3. Adjust the current transform by:
            #   a. Translating by the negative of the centerpoint, to take it out of
            #       the computation.
            #   b. Rotating to adjust the elevation. The axis of rotation is defined
            #       by the camera's right vector. Note that this is done before the
            #       azimuth adjustment because that adjustment will alter the
            #       camera's right vector.
            #   c. Rotating to adjust the azimuth. The axis of rotation is always
            #       the positive Z axis.
            #   d. Translating by the centerpoint, to reorient the camera around
            #           that centerpoint.

            # Step 0: Gather transform components, relative to camera center
            orbit_mat = camera.transform.translated(-center_array)
            position, _rotation, _scale = la.mat_decompose(orbit_mat.T)
            camera_right = np.cross(camera.forward, camera.up)

            # Step 1
            d_azimuth = self._last_canvas_pos[0] - event.canvas_pos[0]
            d_elevation = self._last_canvas_pos[1] - event.canvas_pos[1]

            # Step 2
            e_bound = float(la.vec_angle(position, (0, 0, 1)) * 180 / math.pi)
            if e_bound + d_elevation < 0:
                d_elevation = -e_bound
            if e_bound + d_elevation > 180:
                d_elevation = 180 - e_bound

            # Step 3
            camera.transform = (
                camera.transform.translated(-center_array)  # 3a
                .rotated(d_elevation, camera_right)  # 3b
                .rotated(d_azimuth, self.polar_axis)  # 3c
                .translated(center_array)  # 3d
            )

            handled = True

        # Pan on mouse press with right button
        elif isinstance(event, MousePressEvent) and event.buttons == MouseButton.RIGHT:
            self._pan_ray = event.world_ray

        # Pan on mouse move with right button held
        elif (
            isinstance(event, MouseMoveEvent)
            and event.buttons == MouseButton.RIGHT
            and self._pan_ray is not None
        ):
            dr = np.linalg.norm(camera.transform.map((0, 0, 0))[:3] - center_array)
            old_center = self._pan_ray.origin[:3] + np.multiply(
                dr, self._pan_ray.direction
            )
            new_center = event.world_ray.origin[:3] + np.multiply(
                dr, event.world_ray.direction
            )
            diff = np.subtract(old_center, new_center)
            camera.transform = camera.transform.translated(diff)
            # Update the center
            new_center_array = center_array + diff
            new_center_tuple = (
                float(new_center_array[0]),
                float(new_center_array[1]),
                float(new_center_array[2]),
            )
            self.center = new_center_tuple
            handled = True

        elif isinstance(event, WheelEvent):
            _dx, dy = event.angle_delta
            if dy:
                dr = camera.transform.map((0, 0, 0))[:3] - center_array
                zoom = self._zoom_factor(dy)
                camera.transform = camera.transform.translated(dr * (zoom - 1))
            handled = True

        if isinstance(event, MouseEvent):
            self._last_canvas_pos = event.canvas_pos
        return handled

    def _zoom_factor(self, delta: float) -> float:
        # Magnifier stolen from pygfx
        return 2 ** (-delta * 0.001)
