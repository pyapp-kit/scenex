from __future__ import annotations

import math

import numpy as np
import pylinalg as la

import scenex as snx
from scenex.events.controllers import OrbitController, PanZoomController
from scenex.events.events import MouseButton, MouseEvent, Ray, WheelEvent
from scenex.model._transform import Transform


def test_panzoomcontroller_pan():
    """Tests panning behavior of the PanZoomController."""
    controller = PanZoomController()
    cam = snx.Camera(interactive=True, controller=controller)
    # Simulate mouse press
    press_event = MouseEvent(
        type="press",
        canvas_pos=(0, 0),
        world_ray=Ray((10, 10, 0), (0, 0, -1)),
        buttons=MouseButton.LEFT,
    )
    controller(press_event, cam)
    # Simulate mouse move
    move_event = MouseEvent(
        type="move",
        canvas_pos=(0, 0),
        world_ray=Ray((15, 20, 0), (0, 0, -1)),
        buttons=MouseButton.LEFT,
    )
    controller(move_event, cam)
    # The camera should have moved by (-5, -10)
    expected = Transform().translated((-5, -10))
    np.testing.assert_allclose(cam.transform.root, expected.root)


def test_panzoomcontroller_zoom():
    """Tests zooming behavior of the PanZoomController."""
    controller = PanZoomController()
    cam = snx.Camera(interactive=True)
    cam.set_event_filter(controller)
    # Simulate wheel event
    wheel_event = WheelEvent(
        type="wheel",
        canvas_pos=(0, 0),
        world_ray=Ray((0, 0, 0), (0, 0, -1)),
        buttons=MouseButton.NONE,
        angle_delta=(0, 120),
    )
    before = cam.projection
    controller(wheel_event, cam)
    # The projection should be scaled
    zoom = controller._zoom_factor(wheel_event.angle_delta[1])
    expected = before.scaled((zoom, zoom, 1))
    np.testing.assert_allclose(cam.projection.root, expected.root)


def test_orbitcontroller_orbit():
    """Tests orbiting behavior of the OrbitController."""
    # Camera is along the x axis, looking in the negative x direction at the center
    controller = OrbitController(center=(0, 0, 0))
    cam = snx.Camera(interactive=True, controller=controller)
    # Add cam to the canvas
    canvas = snx.Canvas()
    view = snx.View(camera=cam)
    canvas.views.append(view)
    # Position the camera along the x-axis, looking in the negative x direction at the
    # center
    cam.transform = Transform().rotated(90, (0, 1, 0)).translated((10, 0, 0))
    ray = canvas.to_world((view.layout.width / 2, view.layout.height / 2))
    assert ray is not None
    np.testing.assert_allclose(ray.origin, (10, 0, 0), atol=1e-7)
    np.testing.assert_allclose(ray.direction, (-1, 0, 0), atol=1e-7)

    pos_before = cam.transform.map((0, 0, 0))[:3]
    # Simulate mouse press
    click_pos = (view.layout.width / 2, view.layout.height / 2)
    press_event = MouseEvent(
        type="press",
        canvas_pos=click_pos,
        world_ray=canvas.to_world(click_pos),
        buttons=MouseButton.LEFT,
    )
    controller(press_event, cam)
    # Simulate mouse move (orbit) of one horizontal pixel
    move_pos = (click_pos[0] + 1, click_pos[1])
    move_event = MouseEvent(
        type="move",
        canvas_pos=move_pos,
        world_ray=canvas.to_world(move_pos),
        buttons=MouseButton.LEFT,
    )
    controller(move_event, cam)
    move_pos = (click_pos[0] + 1, click_pos[1] + 1)
    move_event = MouseEvent(
        type="move",
        canvas_pos=move_pos,
        world_ray=canvas.to_world(move_pos),
        buttons=MouseButton.LEFT,
    )
    controller(move_event, cam)
    # Assert camera position conforms to expectation (rotated 1 degree around z axis)
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
    np.testing.assert_allclose(pos_after_exp, pos_after_act)


def test_orbitcontroller_zoom():
    center = (0.0, 0.0, 0.0)
    cam = snx.Camera(interactive=True, transform=Transform().translated((0, 0, 10)))
    controller = OrbitController(center)
    cam.set_event_filter(controller)
    tform_before = cam.transform
    # Simulate wheel event
    wheel_event = WheelEvent(
        type="wheel",
        canvas_pos=(0, 0),
        world_ray=Ray((0, 0, 10), (0, 0, -1)),
        buttons=MouseButton.NONE,
        angle_delta=(0, 120),
    )
    controller(wheel_event, cam)
    # The camera should have moved closer to center
    zoom = controller._zoom_factor(120)
    desired_tform = Transform().translated((0, 0, 10 * zoom))
    np.testing.assert_allclose(cam.transform, desired_tform)

    # Simulate wheel event in other direction
    wheel_event = WheelEvent(
        type="wheel",
        canvas_pos=(0, 0),
        world_ray=Ray((0, 0, 10), (0, 0, -1)),
        buttons=MouseButton.NONE,
        angle_delta=(0, -120),
    )
    controller(wheel_event, cam)
    # The camera should have moved back to the starting point
    zoom = controller._zoom_factor(-120)
    desired_tform = Transform().translated((0, 0, 10))
    np.testing.assert_allclose(cam.transform, tform_before)


def test_orbitcontroller_pan():
    # Camera is along the x axis, looking in the negative x direction at the center
    controller = OrbitController(center=(0, 0, 0))
    cam = snx.Camera(interactive=True, controller=controller)
    # Add cam to the canvas
    canvas = snx.Canvas()
    view = snx.View(camera=cam)
    canvas.views.append(view)
    # Position the camera along the x-axis, looking in the negative x direction at the
    # center
    cam.transform = Transform().rotated(90, (0, 1, 0)).translated((10, 0, 0))
    ray = canvas.to_world((view.layout.width / 2, view.layout.height / 2))
    assert ray is not None
    np.testing.assert_allclose(ray.origin, (10, 0, 0), atol=1e-7)
    np.testing.assert_allclose(ray.direction, (-1, 0, 0), atol=1e-7)
    tform_before = cam.transform
    center_before = controller.center.copy()

    # Simulate right mouse press
    click_pos = (view.layout.width / 2, view.layout.height / 2)
    world_ray_before = canvas.to_world(click_pos)
    press_event = MouseEvent(
        type="press",
        canvas_pos=click_pos,
        world_ray=world_ray_before,
        buttons=MouseButton.RIGHT,
    )
    controller(press_event, cam)
    # Simulate right mouse move (pan)
    click_pos = (click_pos[0], click_pos[1] + view.layout.height // 2)
    world_ray_after = canvas.to_world(click_pos)
    move_event = MouseEvent(
        type="move",
        canvas_pos=click_pos,
        world_ray=world_ray_after,
        buttons=MouseButton.RIGHT,
    )
    controller(move_event, cam)
    # This should move the camera (world_ray_before - world_ray_after), so that the
    # center stays at the same point on the camera plane.
    distance = [
        b - a
        for b, a in zip(world_ray_before.origin, world_ray_after.origin, strict=True)
    ]
    desired_tform = tform_before.translated(distance)
    np.testing.assert_allclose(cam.transform, desired_tform)
    # It should move the orbit center in a similar way
    desired_center = np.asarray(center_before) + distance
    np.testing.assert_allclose(controller.center, desired_center)
