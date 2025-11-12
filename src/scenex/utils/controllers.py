"""Controllers for camera nodes."""

import math

import numpy as np
import pylinalg as la

from scenex.app.events._events import (
    Event,
    MouseButton,
    MouseEvent,
    MouseMoveEvent,
    MousePressEvent,
    Ray,
    WheelEvent,
)
from scenex.model import Camera, Node
from scenex.model._view import View


class OrbitController:
    """
    Orbits a Camera node around a fixed point.

    Rotation direction follows pygfx precedent, where foreground objects (between the
    camera and the center of rotation) move in the direction of mouse movement i.e.
    foreground objects move right when the mouse moves right, and up when the mouse
    moves up.

    Orbit controls define a polar axis (the Z axis in this case), and allow user
    interaction to adjust the camera's angle around the polar axis (azimuth) and angle
    to the polar axis (elevation).

    The left mouse button orbits/rotates the camera.
    Right mouse button: pan.
    Wheel: zoom to point.
    """

    def __init__(self, center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.center = np.array(center, dtype=float)
        self.polar_axis = np.array((0.0, 0.0, 1.0), dtype=float)
        self._last_canvas_pos: tuple[float, float] | None = None
        self._pan_ray: Ray | None = None

    def __call__(self, event: Event, node: Node) -> bool:
        """Handle mouse and wheel events to orbit the camera."""
        # Only operate on INTERACTIVE Camera nodes
        if not isinstance(node, Camera) or not node.interactive:
            return False

        handled = False
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
            orbit_mat = node.transform.translated(-self.center)
            position, _rotation, _scale = la.mat_decompose(orbit_mat.T)
            # TODO: Make this a controller parameter
            camera_polar = (0, 0, 1)
            camera_right = np.cross(node.forward, node.up)

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
            node.transform = (
                node.transform.translated(-self.center)  # 3a
                .rotated(d_elevation, camera_right)  # 3b
                .rotated(d_azimuth, camera_polar)  # 3c
                .translated(self.center)  # 3d
            )

            # Step n+1: Update last position
            self._last_canvas_pos = self._last_canvas_pos
            handled = True

        # Pan on mouse move with right button held
        elif isinstance(event, MousePressEvent) and event.buttons == MouseButton.RIGHT:
            self._pan_ray = event.world_ray

        # Pan on mouse move with right button held
        elif (
            isinstance(event, MouseMoveEvent)
            and event.buttons == MouseButton.RIGHT
            and self._pan_ray is not None
        ):
            dr = np.linalg.norm(node.transform.map((0, 0, 0))[:3] - self.center)
            old_center = self._pan_ray.origin[:3] + np.multiply(
                dr, self._pan_ray.direction
            )
            new_center = event.world_ray.origin[:3] + np.multiply(
                dr, event.world_ray.direction
            )
            diff = np.subtract(old_center, new_center)
            node.transform = node.transform.translated(diff)
            self.center += diff
            handled = True

        elif isinstance(event, WheelEvent):
            _dx, dy = event.angle_delta
            if dy:
                dr = node.transform.map((0, 0, 0))[:3] - self.center
                zoom = self._zoom_factor(dy)
                node.transform = node.transform.translated(dr * (zoom - 1))
            handled = True

        if isinstance(event, MouseEvent):
            self._last_canvas_pos = event.canvas_pos
        return handled

    def _zoom_factor(self, delta: float) -> float:
        # Magnifier stolen from pygfx
        return 2 ** (-delta * 0.001)


class PanZoomController:
    """
    Controller for handling pan and zoom interactions with a Camera node.

    This class enables intuitive mouse-based panning and zooming in a 2D scene.
    It tracks mouse events to allow dragging (panning) the camera view and
    scroll wheel events to zoom in and out, keeping the cursor position fixed
    under the mouse during zoom.
    """

    def __init__(self, lock_x: bool = False, lock_y: bool = False) -> None:
        self._drag_pos: tuple[float, float] | None = None
        self.lock_x = lock_x
        self.lock_y = lock_y

        self._view: View | None = None
        self._old_view_size: tuple[int, int] | None = None

    def maintain_aspect_against(self, view: View | None = None) -> None:
        """Sets up the controller to maintain aspect ratio against the given view."""
        if self._view is not None:
            self._view.layout.events.width.disconnect(self._on_layout_resize)
            self._view.layout.events.height.disconnect(self._on_layout_resize)
        self._view = view
        if self._view is not None:
            self._old_view_size = (
                int(self._view.layout.width),
                int(self._view.layout.height),
            )
            self._view.layout.events.width.connect(self._on_layout_resize)
            self._view.layout.events.height.connect(self._on_layout_resize)

    def _on_layout_resize(self, _: int) -> None:
        """Event filter that scales content proportionally with window size."""
        if self._view is None or self._old_view_size is None:
            return
        new_size = (int(self._view.layout.width), int(self._view.layout.height))
        if any(s == 0 for s in new_size):
            # Canvas is being hidden; skip resize
            return
        width_ratio = self._old_view_size[0] / new_size[0]
        height_ratio = self._old_view_size[1] / new_size[1]
        self._view.camera.projection = self._view.camera.projection.scaled(
            (width_ratio, height_ratio)
        )
        self._old_view_size = new_size

    def __call__(self, event: Event, node: Node) -> bool:
        """Handle mouse and wheel events to pan/zoom the camera."""
        # Only operate on INTERACTIVE Camera nodes
        if not isinstance(node, Camera) or not node.interactive:
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
                node.transform = node.transform.translated((dx, 0))
            dy = self._drag_pos[1] - new_pos[1]
            if not self.lock_y:
                node.transform = node.transform.translated((0, dy))
            handled = True

        # Note that while panning adjusts the camera's transform matrix, zooming
        # adjusts the projection matrix.
        elif isinstance(event, WheelEvent):
            # Zoom while keeping the position under the cursor fixed.
            _dx, dy = event.angle_delta
            if dy:
                # Step 1: Adjust the projection matrix to zoom in or out.
                zoom = self._zoom_factor(dy)
                node.projection = node.projection.scaled(
                    (1 if self.lock_x else zoom, 1 if self.lock_y else zoom, 1.0)
                )

                # Step 2: Adjust the transform matrix to maintain the position
                # under the cursor. The math is largely borrowed from
                # https://github.com/pygfx/pygfx/blob/520af2d5bb2038ec309ef645e4a60d502f00d181/pygfx/controllers/_panzoom.py#L164

                # Find the distance between the world ray and the camera
                zoom_center = np.asarray(event.world_ray.origin)[:2]
                camera_center = np.asarray(node.transform.map((0, 0)))[:2]
                # Compute the world distance before the zoom
                delta_screen1 = zoom_center - camera_center
                # Compute the world distance after the zoom
                delta_screen2 = delta_screen1 * zoom
                # The pan is the difference between the two
                pan = (delta_screen2 - delta_screen1) / zoom
                node.transform = node.transform.translated(
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
