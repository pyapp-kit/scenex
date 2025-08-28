"""Controllers for camera nodes."""

import math

import numpy as np
import pylinalg as la

from scenex.events.events import Event, MouseButton, MouseEvent, WheelEvent
from scenex.model import Camera, Node
from scenex.model._transform import Transform


class OrbitController:
    """
    Controller for orbiting a Camera node around a fixed point.

    Left mouse button: orbit/rotate.
    Right mouse button: pan.
    Wheel: zoom to point.
    """

    def __init__(self, center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self._drag_pos: tuple[float, float] | None = None
        self._center = np.array(center, dtype=float)

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
                self._drag_pos = new_pos

            # Orbit on mouse move with left button held
            elif (
                event.type == "move"
                and MouseButton.LEFT in event.buttons
                and self._drag_pos is not None
            ):
                # Break down the camera transform relative to the orbit center
                orbit_mat = node.transform.translated(-self._center)
                position, rotation, scale = la.mat_decompose(orbit_mat.T)
                # Phi is the angle from the positive z-axis (index 2)
                # Theta is the angle from the positive y-axis (index 1)
                r, phi, theta = la.vec_euclidean_to_spherical(
                    orbit_mat.map((0, 0, 0))[:3]
                )
                # Azimuth is the angle (degrees) from the positive x-axis
                azimuth = (theta * 180 / math.pi) - 90
                # Elevation is the angle (degrees) from the positive z-axis
                elevation = phi * 180 / math.pi
                print(f"r={r}, azimuth={azimuth}, elevation={elevation}")

                # Azimuth angle is horizontal axis, elevation is vertical axis.
                d_azimuth = self._drag_pos[0] - new_pos[0]
                d_elevation = self._drag_pos[1] - new_pos[1]

                new_elevation = max(0, min(180, elevation + d_elevation))

                new_azimuth = azimuth - d_azimuth
                # new_azimuth = azimuth

                node.transform = (
                    Transform()
                    .scaled(scale)
                    .rotated(90, (0, 1, 0))
                    .rotated(90, (1, 0, 0))
                    .translated((r, 0, 0))
                    .rotated(90 - new_elevation, (0, -1, 0))
                    .rotated(new_azimuth, (0, 0, 1))
                    .translated(self._center)
                )

                self._drag_pos = new_pos

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
