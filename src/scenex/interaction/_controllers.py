"""Camera controllers for interactive camera manipulation."""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

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

if TYPE_CHECKING:
    from scenex.app.events import Event
    from scenex.model._view import View


class CameraController(EventedBase):
    """Base class defining how a camera responds to user interaction events.

    A CameraController handles user input (mouse, keyboard, wheel) to manipulate
    camera transforms and projections. Controllers are registered with a
    CanvasInteractor for a specific view and automatically receive events routed
    to that view.

    Event handlers should return True if they fully handled the event (stopping
    further propagation) or False if other handlers should continue processing.

    Examples
    --------
    Register a controller with a CanvasInteractor::

        ci = CanvasInteractor(canvas)
        ci.set_controller(view, PanZoom())

    See Also
    --------
    PanZoom : 2D pan and zoom controller
    Orbit : 3D orbit controller
    CanvasInteractor : Coordinator that routes events to controllers
    """

    @abstractmethod
    def handle_event(self, event: Event, view: View) -> bool:
        """Handle a user interaction event to control the camera.

        Parameters
        ----------
        event : Event
            The input event to handle.
        view : View
            The view containing the camera to manipulate.

        Returns
        -------
        bool
            True if the event was fully handled and should not propagate,
            False otherwise.

        Notes
        -----
        A ``View`` is passed rather than a ``Camera`` directly because controllers
        need ``view.to_ray()`` to unproject screen-space positions into world space.
        """
        raise NotImplementedError


class PanZoom(CameraController):
    """2D pan and zoom controller for orthographic views.

    - **Panning** (left mouse drag): Modifies camera.transform to translate the camera.
    - **Zooming** (mouse wheel): Modifies camera.projection to scale the view, then
      adjusts camera.transform to keep zoom centered on the cursor position.

    Attributes
    ----------
    lock_x : bool
        If True, prevent horizontal panning and zooming.
    lock_y : bool
        If True, prevent vertical panning and zooming.

    Examples
    --------
    Register with CanvasInteractor::

        ci = CanvasInteractor(canvas)
        ci.set_controller(view, PanZoom())

    See Also
    --------
    Orbit : 3D orbit controller for perspective views
    CameraController : Base class for camera controllers
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

    _drag_pos: tuple[float, float] | None = PrivateAttr(default=None)

    def handle_event(self, event: Event, view: View) -> bool:
        """Handle mouse and wheel events to pan/zoom the camera."""
        handled = False

        if not isinstance(event, MouseEvent):
            return False
        if (ray := view.to_ray(event.pos)) is None:
            return False

        if isinstance(event, MousePressEvent) and MouseButton.LEFT in event.buttons:
            self._drag_pos = ray.origin[:2]
        elif (
            isinstance(event, MouseMoveEvent)
            and MouseButton.LEFT in event.buttons
            and self._drag_pos
        ):
            new_pos = ray.origin[:2]
            dx = self._drag_pos[0] - new_pos[0]
            if not self.lock_x:
                view.camera.transform = view.camera.transform.translated((dx, 0))
            dy = self._drag_pos[1] - new_pos[1]
            if not self.lock_y:
                view.camera.transform = view.camera.transform.translated((0, dy))
            handled = True

        elif isinstance(event, WheelEvent):
            _, dy = event.angle_delta
            if dy:
                zoom = self._zoom_factor(dy)
                view.camera.projection = view.camera.projection.scaled(
                    (1 if self.lock_x else zoom, 1 if self.lock_y else zoom, 1.0)
                )
                zoom_center = np.asarray(ray.origin)[:2]
                camera_center = np.asarray(view.camera.transform.map((0, 0)))[:2]
                delta_screen1 = zoom_center - camera_center
                delta_screen2 = delta_screen1 * zoom
                pan = (delta_screen2 - delta_screen1) / zoom
                view.camera.transform = view.camera.transform.translated(
                    (
                        pan[0] if not self.lock_x else 0,
                        pan[1] if not self.lock_y else 0,
                    )
                )
                handled = True

        return handled

    def _zoom_factor(self, delta: float) -> float:
        return 2 ** (delta * 0.001)


class Orbit(CameraController):
    """3D orbit controller for rotating around a focal point.

    - **Left drag**: Orbit/rotate the camera around the center point
    - **Right drag**: Pan the orbit center (translates the focal point)
    - **Mouse wheel**: Zoom toward/away from center (change radius)

    Attributes
    ----------
    center : tuple[float, float, float]
        The point in 3D space around which the camera orbits.
    polar_axis : tuple[float, float, float]
        The axis defining the "up" direction for orbit calculations.

    Examples
    --------
    Register with CanvasInteractor::

        ci = CanvasInteractor(canvas)
        ci.set_controller(view, Orbit(center=(0, 0, 0)))

    See Also
    --------
    PanZoom : 2D pan and zoom controller for orthographic views
    CameraController : Base class for camera controllers
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

    _last_canvas_pos: tuple[float, float] | None = PrivateAttr(default=None)
    _pan_ray: Any = PrivateAttr(default=None)

    def handle_event(self, event: Event, view: View) -> bool:
        """Handle mouse and wheel events to orbit the camera."""
        handled = False
        center_array = np.asarray(self.center)

        if not isinstance(event, MouseEvent):
            return False
        if (ray := view.to_ray(event.pos)) is None:
            return False

        if (
            isinstance(event, MouseMoveEvent)
            and event.buttons == MouseButton.LEFT
            and self._last_canvas_pos is not None
        ):
            orbit_mat = view.camera.transform.translated(-center_array)
            position = la.mat_decompose(orbit_mat.T)[0]
            camera_right = np.cross(view.camera.forward, view.camera.up)

            d_azimuth = self._last_canvas_pos[0] - event.pos[0]
            d_elevation = self._last_canvas_pos[1] - event.pos[1]

            e_bound = float(la.vec_angle(position, (0, 0, 1)) * 180 / math.pi)
            if e_bound + d_elevation < 0:
                d_elevation = -e_bound
            if e_bound + d_elevation > 180:
                d_elevation = 180 - e_bound

            view.camera.transform = (
                view.camera.transform.translated(-center_array)
                .rotated(d_elevation, camera_right)
                .rotated(d_azimuth, self.polar_axis)
                .translated(center_array)
            )
            handled = True

        elif isinstance(event, MousePressEvent) and event.buttons == MouseButton.RIGHT:
            self._pan_ray = ray

        elif (
            isinstance(event, MouseMoveEvent)
            and event.buttons == MouseButton.RIGHT
            and self._pan_ray is not None
        ):
            dr = np.linalg.norm(view.camera.transform.map((0, 0, 0))[:3] - center_array)
            old_center = self._pan_ray.origin[:3] + np.multiply(
                dr, self._pan_ray.direction
            )
            new_center = ray.origin[:3] + np.multiply(dr, ray.direction)
            diff = np.subtract(old_center, new_center)
            view.camera.transform = view.camera.transform.translated(diff)
            new_center_array = center_array + diff
            self.center = (
                float(new_center_array[0]),
                float(new_center_array[1]),
                float(new_center_array[2]),
            )
            handled = True

        elif isinstance(event, WheelEvent):
            _, dy = event.angle_delta
            if dy:
                dr = view.camera.transform.map((0, 0, 0))[:3] - center_array
                zoom = self._zoom_factor(dy)
                view.camera.transform = view.camera.transform.translated(
                    dr * (zoom - 1)
                )
            handled = True

        if isinstance(event, MouseEvent):
            self._last_canvas_pos = event.pos
        return handled

    def _zoom_factor(self, delta: float) -> float:
        return 2 ** (-delta * 0.001)


AnyController = PanZoom | Orbit | None
