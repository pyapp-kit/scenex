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
        self._azimuth = 0.0
        self._elevation = 0.0
        self._radius = 5.0  # Default distance from center

    def __call__(self, event: Event, node: Node) -> bool:
        assert isinstance(node, Camera)
        handled = False

        if isinstance(event, MouseEvent):
            new_pos = event.canvas_pos

            # Start orbit on left mouse press
            if event.type == "press" and MouseButton.LEFT in event.buttons:
                self._drag_pos = new_pos

            # Orbit on mouse move with left button held
            elif (
                event.type == "move"
                and MouseButton.LEFT in event.buttons
                and self._drag_pos
            ):
                # Azimuth angle is horizontal axis, elevation is vertical axis.
                delta_azimuth = self._drag_pos[0] - new_pos[0]
                # delta_elevation = self._drag_pos[1] - new_pos[1]

                up = la.vec_normalize(
                    node.projection.map((0, 1)) - node.projection.map((0, 0))
                )[:3]

                # quat_azimuth = la.quat_from_axis_angle(up, -delta_azimuth * 0.005)
                # Suppose you want to rotate X degrees. Pass ()
                print(delta_azimuth)
                quat_azimuth = la.quat_from_axis_angle(
                    up, -delta_azimuth * math.pi / 180
                )
                # quat_elevation = la.quat_from_axis_angle((1, 0, 0), -delta_elevation)

                position = la.vec_transform_quat(
                    node.transform.root[3, :3],
                    quat_azimuth,
                )

                node.transform = Transform().translated(position)
                node.projection = node.projection @ la.mat_from_axis_angle(
                    up, delta_azimuth * math.pi / 180
                )
                self._drag_pos = new_pos
                # print(position)

                # position = camera_state["position"]
                # rotation = camera_state["rotation"]
                # up = camera_state["reference_up"]

                # # Where is the camera looking at right now
                # forward = la.vec_transform_quat((0, 0, -1), rotation)

                # # # Get a reference vector, that is orthogonal to up, in a deterministic way.
                # # # Might need this if we ever want the azimuth
                # # aligned_up = _get_axis_aligned_up_vector(up)
                # # orthogonal_vec = np.cross(up, np.roll(aligned_up, 1))

                # # Get current elevation, so we can clip it.
                # # We don't need the azimuth. When we do, it'd need more care to get a proper 0..2pi range
                # elevation = la.vec_angle(forward, up) - 0.5 * np.pi

                # # Apply boundaries to the elevation
                # new_elevation = elevation + delta_elevation
                # bounds = -89 * np.pi / 180, 89 * np.pi / 180
                # if new_elevation < bounds[0]:
                #     delta_elevation = bounds[0] - elevation
                # elif new_elevation > bounds[1]:
                #     delta_elevation = bounds[1] - elevation

                # r_azimuth = la.quat_from_axis_angle(up, -delta_azimuth)
                # r_elevation = la.quat_from_axis_angle((1, 0, 0), -delta_elevation)

                # # Get rotations
                # rot1 = rotation
                # rot2 = la.quat_mul(r_azimuth, la.quat_mul(rot1, r_elevation))

                # # Calculate new position
                # pos1 = position
                # pos2target1 = self._get_target_vec(camera_state, rotation=rot1)
                # pos2target2 = self._get_target_vec(camera_state, rotation=rot2)
                # pos2 = pos1 + pos2target1 - pos2target2

            # # Pan with right mouse button
            # elif (
            #     event.type == "move"
            #     and MouseButton.RIGHT in event.buttons
            #     and self._drag_pos
            # ):
            #     dx = self._drag_pos[0] - new_pos[0]
            #     dy = self._drag_pos[1] - new_pos[1]
            #     node.transform = node.transform.translated((dx, dy))
            #     handled = True

            # elif event.type == "press" and MouseButton.RIGHT in event.buttons:
            #     self._drag_pos = new_pos

            # elif event.type == "release" and MouseButton.RIGHT not in event.buttons:
            #     self._drag_pos = None

            # Zoom with wheel
            elif isinstance(event, WheelEvent):
                _dx, dy = event.angle_delta
                if dy:
                    zoom = 2 ** (dy * 0.001)
                    self._radius /= zoom
                    self._orbit_camera(node)
                    handled = True

        return handled

    def _orbit_camera(self, node: Camera) -> None:
        # Calculate camera position in spherical coordinates
        r = self._radius
        az = self._azimuth
        el = self._elevation
        # Spherical to Cartesian
        x = r * math.cos(el) * math.sin(az)
        y = r * math.sin(el)
        z = r * math.cos(el) * math.cos(az)
        pos = self._target + np.array([x, y, z])

        # Look-at matrix: camera at pos, looking at target
        up = np.array([0, 1, 0], dtype=float)
        fwd = self._target - pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(up, fwd)
        right /= np.linalg.norm(right)
        up = np.cross(fwd, right)

        # Build rotation matrix
        rot = np.eye(4)
        rot[:3, 0] = right
        rot[:3, 1] = up
        rot[:3, 2] = fwd

        # Build transform matrix
        node.transform = node.transform.translated(pos).rotated(rot)


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
