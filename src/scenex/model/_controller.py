"""Base class for camera controllers."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field, PrivateAttr

from scenex.model._base import EventedBase

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scenex.app.events._events import Event

    from ._nodes.camera import Camera


class Controller(EventedBase):
    """
    Base class for camera controllers.

    Controllers are declarative, pydantic-based classes that define how a camera
    responds to user input events. They are attached to a Camera model and handle
    events such as mouse clicks, drags, and wheel scrolls.

    Unlike the imperative controller pattern, these controllers are part of the
    scene graph model and can be serialized, modified, and observed via events.
    """

    @abstractmethod
    def handle_event(self, event: Event, camera: Camera) -> bool:
        """
        Handle an event for the given camera.

        Parameters
        ----------
        event : Event
            The input event to handle (mouse, keyboard, wheel, etc.)
        camera : Camera
            The camera node to manipulate

        Returns
        -------
        bool
            True if the event was handled and should not propagate further,
            False otherwise.
        """
        raise NotImplementedError


class PanZoomController(Controller):
    """
    Controller for handling pan and zoom interactions with a Camera node.

    This controller enables intuitive mouse-based panning and zooming in a 2D scene.
    It tracks mouse events to allow dragging (panning) the camera view and
    scroll wheel events to zoom in and out, keeping the cursor position fixed
    under the mouse during zoom.

    Attributes
    ----------
    lock_x : bool
        If True, prevent horizontal panning and zooming.
    lock_y : bool
        If True, prevent vertical panning and zooming.
    """

    lock_x: bool = Field(
        default=False,
        description="If True, prevent horizontal panning and zooming.",
    )
    lock_y: bool = Field(
        default=False,
        description="If True, prevent vertical panning and zooming.",
    )

    # Private state for tracking interactions
    _drag_pos: tuple[float, float] | None = PrivateAttr(default=None)

    def handle_event(self, event: Event, camera: Camera) -> bool:
        """Handle mouse and wheel events to pan/zoom the camera."""
        from scenex.app.events._events import (
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


class OrbitController(Controller):
    """
    Orbits a Camera node around a fixed point.

    Rotation direction follows pygfx precedent, where foreground objects (between the
    camera and the center of rotation) move in the direction of mouse movement i.e.
    foreground objects move right when the mouse moves right, and up when the mouse
    moves up.

    Orbit controls define a polar axis (the Z axis by default), and allow user
    interaction to adjust the camera's angle around the polar axis (azimuth) and angle
    to the polar axis (elevation).

    Attributes
    ----------
    center : tuple[float, float, float]
        The point in 3D space around which the camera orbits.
    polar_axis : tuple[float, float, float]
        The axis defining the "up" direction for orbit calculations.

    Interactions
    ------------
    - Left mouse button: orbit/rotate the camera
    - Right mouse button: pan the orbit center
    - Mouse wheel: zoom toward/away from center
    """

    center: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="The point in 3D space around which the camera orbits.",
    )
    polar_axis: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 1.0),
        description='The axis defining the "up" direction for orbit calculations.',
    )

    # Private state for tracking interactions
    _last_canvas_pos: tuple[float, float] | None = PrivateAttr(default=None)
    _pan_ray: Any = PrivateAttr(default=None)  # Ray type

    def model_post_init(self, __context: Any) -> None:
        """Ensure center and polar_axis are numpy arrays."""
        super().model_post_init(__context)
        # Convert to numpy arrays for efficient computation
        object.__setattr__(self, "_center_array", np.array(self.center, dtype=float))
        object.__setattr__(
            self, "_polar_axis_array", np.array(self.polar_axis, dtype=float)
        )

    def handle_event(self, event: Event, camera: Camera) -> bool:
        """Handle mouse and wheel events to orbit the camera."""
        import math

        import pylinalg as la

        from scenex.app.events._events import (
            MouseButton,
            MouseEvent,
            MouseMoveEvent,
            MousePressEvent,
            WheelEvent,
        )

        if not camera.interactive:
            return False

        handled = False
        center_array: NDArray[np.floating[Any]] = object.__getattribute__(
            self, "_center_array"
        )

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
            # TODO: Make this a controller parameter
            camera_polar = (0, 0, 1)
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
                .rotated(d_azimuth, camera_polar)  # 3c
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
            object.__setattr__(self, "_center_array", new_center_array)
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
