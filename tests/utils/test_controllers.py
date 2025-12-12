from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pylinalg as la
import pytest

import scenex as snx
from scenex.app.events import (
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    Ray,
    WheelEvent,
)
from scenex.model._transform import Transform


def _validate_ray(maybe_ray: Ray | None) -> Ray:
    assert maybe_ray is not None
    return maybe_ray


def test_panzoomcontroller_pan() -> None:
    """Tests panning behavior of the PanZoom interaction strategy."""
    controller = snx.PanZoom()
    cam = snx.Camera(interactive=True, controller=controller)
    # Simulate mouse press
    press_event = MousePressEvent(
        canvas_pos=(0, 0),
        world_ray=Ray((10, 10, 0), (0, 0, -1), source=MagicMock(spec=snx.View)),
        buttons=MouseButton.LEFT,
    )
    controller.handle_event(press_event, cam)
    # Simulate mouse move
    move_event = MouseMoveEvent(
        canvas_pos=(0, 0),
        world_ray=Ray((15, 20, 0), (0, 0, -1), source=MagicMock(spec=snx.View)),
        buttons=MouseButton.LEFT,
    )
    controller.handle_event(move_event, cam)
    # The camera should have moved by (-5, -10)
    expected = Transform().translated((-5, -10))
    np.testing.assert_allclose(cam.transform.root, expected.root)


def test_panzoomcontroller_zoom() -> None:
    """Tests zooming behavior of the PanZoomController."""
    controller = snx.PanZoom()
    cam = snx.Camera(interactive=True, controller=controller)
    # Simulate wheel event
    wheel_event = WheelEvent(
        canvas_pos=(0, 0),
        world_ray=Ray((0, 0, 0), (0, 0, -1), source=MagicMock(spec=snx.View)),
        buttons=MouseButton.NONE,
        angle_delta=(0, 120),
    )
    before = cam.projection
    controller.handle_event(wheel_event, cam)
    # The projection should be scaled
    zoom = controller._zoom_factor(wheel_event.angle_delta[1])
    expected = before.scaled((zoom, zoom, 1))
    np.testing.assert_allclose(cam.projection.root, expected.root)


def test_panzoomcontroller_maintain_aspect() -> None:
    """Tests PanZoomController's ability to maintain its aspect."""
    # TODO: This test is disabled because maintain_aspect_against has not yet been
    # implemented in the new declarative controller system.
    # The old imperative controller code has been removed.
    pytest.skip(
        "maintain_aspect_against not yet implemented in declarative controllers"
    )


def test_orbitcontroller_orbit() -> None:
    """Tests orbiting behavior of the OrbitController."""
    # Camera is along the x axis, looking in the negative x direction at the center
    controller = snx.Orbit(center=(0, 0, 0))
    cam = snx.Camera(interactive=True, controller=controller)
    # Add cam to the canvas
    canvas = snx.Canvas()
    view = snx.View(camera=cam)
    canvas.grid.add(view)
    # Position the camera along the x-axis, looking in the negative x direction at the
    # center
    cam.transform = Transform().translated((10, 0, 0))
    cam.look_at((0, 0, 0), up=(0, 0, 1))
    ray = canvas.to_world((view.layout.width / 2, view.layout.height / 2))
    assert ray is not None
    np.testing.assert_allclose(ray.origin, (10, 0, 0), atol=1e-7)
    np.testing.assert_allclose(ray.direction, (-1, 0, 0), atol=1e-7)

    pos_before = cam.transform.map((0, 0, 0))[:3]
    # Simulate mouse press
    click_pos = (view.layout.width / 2, view.layout.height / 2)
    press_event = MousePressEvent(
        canvas_pos=click_pos,
        world_ray=_validate_ray(canvas.to_world(click_pos)),
        buttons=MouseButton.LEFT,
    )
    controller.handle_event(press_event, cam)
    # Simulate mouse move (orbit) of one horizontal pixel and one vertical pixel
    move_pos = (click_pos[0] + 1, click_pos[1] + 1)
    move_event = MouseMoveEvent(
        canvas_pos=move_pos,
        world_ray=_validate_ray(canvas.to_world(move_pos)),
        buttons=MouseButton.LEFT,
    )
    controller.handle_event(move_event, cam)
    # Assert camera position conforms to expectation
    # (one degree around y axis and one degree around z axis)
    pos_after_exp = la.vec_transform_quat(
        pos_before,
        la.quat_mul(
            # Increase azimuth 1 degree
            la.quat_from_axis_angle((0, 0, -1), math.pi / 180),
            # Increase elevation 1 degree
            la.quat_from_axis_angle((0, -1, 0), math.pi / 180),
        ),
    )
    pos_after_act = cam.transform.map((0, 0, 0))[:3]
    np.testing.assert_allclose(pos_after_act, pos_after_exp)


def test_orbitcontroller_zoom() -> None:
    center = (0.0, 0.0, 0.0)
    controller = snx.Orbit(center=center)
    cam = snx.Camera(
        interactive=True,
        transform=Transform().translated((0, 0, 10)),
        controller=controller,
    )
    tform_before = cam.transform
    # Simulate wheel event
    wheel_event = WheelEvent(
        canvas_pos=(0, 0),
        world_ray=Ray((0, 0, 10), (0, 0, -1), source=MagicMock(spec=snx.View)),
        buttons=MouseButton.NONE,
        angle_delta=(0, 120),
    )
    controller.handle_event(wheel_event, cam)
    # The camera should have moved closer to center
    zoom = controller._zoom_factor(120)
    desired_tform = Transform().translated((0, 0, 10 * zoom))
    np.testing.assert_allclose(cam.transform, desired_tform)

    # Simulate wheel event in other direction
    wheel_event = WheelEvent(
        canvas_pos=(0, 0),
        world_ray=Ray((0, 0, 10), (0, 0, -1), source=MagicMock(spec=snx.View)),
        buttons=MouseButton.NONE,
        angle_delta=(0, -120),
    )
    controller.handle_event(wheel_event, cam)
    # The camera should have moved back to the starting point
    zoom = controller._zoom_factor(-120)
    desired_tform = Transform().translated((0, 0, 10))
    np.testing.assert_allclose(cam.transform, tform_before)


def test_orbitcontroller_pan() -> None:
    # Camera is along the x axis, looking in the negative x direction at the center
    controller = snx.Orbit(center=(0, 0, 0))
    cam = snx.Camera(interactive=True, controller=controller)
    # Add cam to the canvas
    canvas = snx.Canvas()
    view = snx.View(camera=cam)
    canvas.grid.add(view)
    # Position the camera along the x-axis, looking in the negative x direction at the
    # center
    cam.transform = Transform().rotated(90, (0, 1, 0)).translated((10, 0, 0))
    ray = canvas.to_world((view.layout.width / 2, view.layout.height / 2))
    assert ray is not None
    np.testing.assert_allclose(ray.origin, (10, 0, 0), atol=1e-7)
    np.testing.assert_allclose(ray.direction, (-1, 0, 0), atol=1e-7)
    tform_before = cam.transform
    center_before = np.array(controller.center)

    # Simulate right mouse press
    click_pos = (view.layout.width / 2, view.layout.height / 2)
    world_ray_before = canvas.to_world(click_pos)
    assert world_ray_before is not None
    press_event = MousePressEvent(
        canvas_pos=click_pos,
        world_ray=world_ray_before,
        buttons=MouseButton.RIGHT,
    )
    controller.handle_event(press_event, cam)
    # Simulate right mouse move (pan)
    click_pos = (click_pos[0], click_pos[1] + view.layout.height // 2)
    world_ray_after = canvas.to_world(click_pos)
    assert world_ray_after is not None
    move_event = MouseMoveEvent(
        canvas_pos=click_pos,
        world_ray=world_ray_after,
        buttons=MouseButton.RIGHT,
    )
    controller.handle_event(move_event, cam)
    # This should move the camera (world_ray_before - world_ray_after), so that the
    # center stays at the same point on the camera plane.
    distance = [
        b - a
        for b, a in zip(world_ray_before.origin, world_ray_after.origin, strict=True)
    ]
    desired_tform = tform_before.translated(distance)
    np.testing.assert_allclose(cam.transform, desired_tform)
    # It should move the orbit center in a similar way
    desired_center = center_before + distance
    np.testing.assert_allclose(controller.center, desired_center)
