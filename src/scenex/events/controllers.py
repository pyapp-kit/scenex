"""Controllers for camera nodes."""

import math

import numpy as np
import pylinalg as la

from scenex.events.events import Event, MouseButton, MouseEvent, WheelEvent
from scenex.model import Camera, Node


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
        self._last_pos: tuple[float, float] | None = None

    def __call__(self, event: Event, node: Node) -> bool:
        """Handle mouse and wheel events to orbit the camera."""
        # TODO: Rigorous documentation
        assert isinstance(node, Camera)
        handled = False

        if isinstance(event, MouseEvent):
            # TODO: Pan with right click
            # TODO: Zoom with wheel
            new_pos = event.canvas_pos

            # Start orbit on left mouse press
            if event.type == "press" and MouseButton.LEFT in event.buttons:
                self._last_pos = new_pos

            # Orbit on mouse move with left button held
            elif (
                event.type == "move"
                and MouseButton.LEFT in event.buttons
                and self._last_pos is not None
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
                position, rotation, _scale = la.mat_decompose(orbit_mat.T)
                camera_right = la.vec_transform_quat((1, 0, 0), rotation)
                # TODO: Make this a controller parameter
                camera_polar = (0, 0, 1)

                # Step 1
                d_azimuth = self._last_pos[0] - new_pos[0]
                d_elevation = float(self._last_pos[1] - new_pos[1])

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
                self._last_pos = new_pos

        return handled


class PanZoomController:
    """
    Controller for handling pan and zoom interactions with a Camera node.

    This class enables intuitive mouse-based panning and zooming in a 2D scene.
    It tracks mouse events to allow dragging (panning) the camera view and
    scroll wheel events to zoom in and out, keeping the cursor position fixed
    under the mouse during zoom.
    """

    def __init__(self) -> None:
        self._drag_pos: tuple[float, float] | None = None

    def __call__(self, event: Event, node: Node) -> bool:
        """Handle mouse and wheel events to pan/zoom the camera."""
        assert isinstance(node, Camera)
        handled = False

        # FIXME: Probably doesn't work outside of panzoom camera
        if isinstance(event, MouseEvent):
            new_pos = event.world_ray.origin[:2]

            # Panning involves keeping a particular position underneath the cursor.
            # That position is recorded on a left mouse button press.
            if event.type == "press" and MouseButton.LEFT in event.buttons:
                self._drag_pos = new_pos
            # Every time the cursor is moved, until the left mouse button is released,
            # We translate the camera such that the position is back under the cursor
            # (i.e. under the world ray origin)
            elif (
                event.type == "move"
                and MouseButton.LEFT in event.buttons
                and self._drag_pos
            ):
                dx = self._drag_pos[0] - new_pos[0]
                dy = self._drag_pos[1] - new_pos[1]
                node.transform = node.transform.translated((dx, dy))
                handled = True

            elif isinstance(event, WheelEvent):
                # Zoom while keeping the position under the cursor fixed.
                _dx, dy = event.angle_delta
                if dy:
                    # Step 1: Adjust the projection matrix to zoom in or out.
                    zoom = 2 ** (dy * 0.001)  # Magnifier stolen from pygfx
                    node.projection = node.projection.scaled((zoom, zoom, 1.0))

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
                    node.transform = node.transform.translated(pan)
                    handled = True

        return handled
