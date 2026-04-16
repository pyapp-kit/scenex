import math
from unittest.mock import MagicMock

import numpy as np
import pylinalg as la

import scenex as snx
from scenex.app.events import (
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    Ray,
    WheelEvent,
)


def test_camera_forward_property() -> None:
    cam = snx.Camera(transform=snx.Transform())
    # Default forward should be (0, 0, -1)
    np.testing.assert_allclose(cam.forward, (0, 0, -1), atol=1e-6)
    # Set forward to (0, 0, 1)
    cam.forward = (0, 0, 1)
    # Test that the camera transform maps (0, 0, -1) to our new forward
    new_fwd = cam.transform.map((0, 0, -1))[:3]
    new_fwd /= np.linalg.norm(new_fwd)
    np.testing.assert_allclose(new_fwd, (0, 0, 1), atol=1e-6)

    # Test that setting camera forward to the same value does not change anything
    # (cover divide-by-zero cases)
    cam.forward = (0, 0, 1)
    # Test that the camera transform STILL maps (0, 0, -1) to our new forward
    new_fwd = cam.transform.map((0, 0, -1))[:3]
    new_fwd /= np.linalg.norm(new_fwd)
    np.testing.assert_allclose(new_fwd, (0, 0, 1), atol=1e-6)


def test_camera_up_property() -> None:
    cam = snx.Camera(transform=snx.Transform())
    # Default up should be (0, 1, 0)
    np.testing.assert_allclose(cam.up, (0, 1, 0), atol=1e-6)
    # Set up to (1, 0, 0)
    cam.up = (1, 0, 0)
    # Test that the camera transform maps (0, 1, 0) to our new up
    new_fwd = cam.transform.map((0, 1, 0))[:3]
    new_fwd /= np.linalg.norm(new_fwd)
    np.testing.assert_allclose(new_fwd, (1, 0, 0), atol=1e-6)
    # Test that the camera forward is still the default
    new_fwd = cam.transform.map((0, 0, -1))[:3]
    new_fwd /= np.linalg.norm(new_fwd)
    np.testing.assert_allclose(new_fwd, (0, 0, -1), atol=1e-6)

    # Test that setting camera up to the same value does not change anything
    # (cover divide-by-zero cases)
    cam.up = (1, 0, 0)
    # Test that the camera transform STILL maps (0, 1, 0) to our new up
    new_fwd = cam.transform.map((0, 1, 0))[:3]
    new_fwd /= np.linalg.norm(new_fwd)
    np.testing.assert_allclose(new_fwd, (1, 0, 0), atol=1e-6)


def test_camera_look_at() -> None:
    cam = snx.Camera(transform=snx.Transform())
    # Look at (0, 0, 1) with up (0, 0, 1)
    cam.look_at((1, 0, 0), up=(0, 0, 1))
    # Forward should be (1, 0, 0) - (0, 0, 0) = (1, 0, 0)
    np.testing.assert_allclose(cam.forward, (1, 0, 0), atol=1e-6)
    # Up should be (0, 0, 1) - (0, 0, 0) = (0, 0, 1)
    np.testing.assert_allclose(cam.up, (0, 0, 1), atol=1e-6)


def _validate_ray(maybe_ray: Ray | None) -> Ray:
    assert maybe_ray is not None
    return maybe_ray


def test_panzoom_mouse() -> None:
    """Tests panning behavior of PanZoom."""
    interaction = snx.PanZoom()
    cam = snx.Camera(interactive=True, controller=interaction)
    # Simulate mouse press
    press_event = MousePressEvent(
        canvas_pos=(0, 0),
        world_ray=Ray((10, 10, 0), (0, 0, -1), source=MagicMock(spec=snx.View)),
        buttons=MouseButton.LEFT,
    )
    interaction.handle_event(press_event, cam)
    # Simulate mouse move
    move_event = MouseMoveEvent(
        canvas_pos=(0, 0),
        world_ray=Ray((15, 20, 0), (0, 0, -1), source=MagicMock(spec=snx.View)),
        buttons=MouseButton.LEFT,
    )
    interaction.handle_event(move_event, cam)
    # The camera should have moved by (-5, -10)
    expected = snx.Transform().translated((-5, -10))
    np.testing.assert_allclose(cam.transform.root, expected.root)


def test_panzoom_scroll() -> None:
    """Tests zooming behavior of PanZoom."""
    interaction = snx.PanZoom()
    cam = snx.Camera(interactive=True, controller=interaction)
    # Simulate wheel event
    wheel_event = WheelEvent(
        canvas_pos=(0, 0),
        world_ray=Ray((0, 0, 0), (0, 0, -1), source=MagicMock(spec=snx.View)),
        buttons=MouseButton.NONE,
        angle_delta=(0, 120),
    )
    before = cam.projection
    interaction.handle_event(wheel_event, cam)
    # The projection should be scaled
    zoom = interaction._zoom_factor(wheel_event.angle_delta[1])
    expected = before.scaled((zoom, zoom, 1))
    np.testing.assert_allclose(cam.projection.root, expected.root)


def test_panzoom_zoom() -> None:
    """Tests zooming via the public zoom() API without a center."""
    interaction = snx.PanZoom()
    cam = snx.Camera(interactive=True, controller=interaction)
    factor = 0.5
    before = cam.projection
    interaction.zoom(cam, factor)
    expected = before.scaled((factor, factor, 1))
    np.testing.assert_allclose(cam.projection.root, expected.root)
    # No center provided — transform should be unchanged
    np.testing.assert_allclose(cam.transform.root, snx.Transform().root)


def test_panzoom_zoom_with_center() -> None:
    """Tests that zoom() applies a compensating translation to keep center fixed."""
    interaction = snx.PanZoom()
    cam = snx.Camera(interactive=True, controller=interaction)
    factor = 0.5
    center = (0.5, 0.3, 0.0)
    interaction.zoom(cam, factor, center=center)
    # Projection should be scaled
    expected_proj = snx.Transform().scaled((factor, factor, -1))
    np.testing.assert_allclose(cam.projection.root, expected_proj.root)
    # Transform should have been panned by the compensating amount
    zoom_center = np.array(center[:2])
    camera_center = np.zeros(2)  # camera was at origin before zoom
    delta_screen1 = zoom_center - camera_center
    delta_screen2 = delta_screen1 * factor
    pan = (delta_screen2 - delta_screen1) / factor
    expected_transform = snx.Transform().translated((pan[0], pan[1]))
    np.testing.assert_allclose(cam.transform.root, expected_transform.root)


def test_panzoom_zoom_lock_x() -> None:
    """Tests that lock_x prevents x-axis scaling and x-axis panning."""
    interaction = snx.PanZoom(lock_x=True)
    cam = snx.Camera(interactive=True, controller=interaction)
    factor = 0.5
    center = (0.5, 0.3, 0.0)
    before_proj = cam.projection
    interaction.zoom(cam, factor, center=center)
    # X axis of projection should be unchanged, Y axis should be scaled
    expected_proj = before_proj.scaled((1, factor, 1))
    np.testing.assert_allclose(cam.projection.root, expected_proj.root)
    # X component of the pan should be zero
    np.testing.assert_allclose(cam.transform.root[3, 0], 0.0)


def test_panzoom_zoom_lock_y() -> None:
    """Tests that lock_y prevents y-axis scaling and y-axis panning."""
    interaction = snx.PanZoom(lock_y=True)
    cam = snx.Camera(interactive=True, controller=interaction)
    factor = 0.5
    center = (0.5, 0.3, 0.0)
    before_proj = cam.projection
    interaction.zoom(cam, factor, center=center)
    # Y axis of projection should be unchanged, X axis should be scaled
    expected_proj = before_proj.scaled((factor, 1, 1))
    np.testing.assert_allclose(cam.projection.root, expected_proj.root)
    # Y component of the pan should be zero
    np.testing.assert_allclose(cam.transform.root[3, 1], 0.0)


def test_orbit_mouse_left() -> None:
    """Tests orbiting behavior of Orbit."""
    # Camera is along the x axis, looking in the negative x direction at the center
    interaction = snx.Orbit(center=(0, 0, 0))
    cam = snx.Camera(interactive=True, controller=interaction)
    # Add cam to the canvas
    view = snx.View(camera=cam)
    canvas = snx.Canvas(views=[view])
    # Position the camera along the x-axis, looking in the negative x direction at the
    # center
    cam.transform = snx.Transform().translated((10, 0, 0))
    cam.look_at((0, 0, 0), up=(0, 0, 1))
    _, _, w, h = canvas.rect_for(view)
    ray = canvas.to_world((w / 2, h / 2))
    assert ray is not None
    np.testing.assert_allclose(ray.origin, (10, 0, 0), atol=1e-7)
    np.testing.assert_allclose(ray.direction, (-1, 0, 0), atol=1e-7)

    pos_before = cam.transform.map((0, 0, 0))[:3]
    # Simulate mouse press
    click_pos = (w / 2, h / 2)
    press_event = MousePressEvent(
        canvas_pos=click_pos,
        world_ray=_validate_ray(canvas.to_world(click_pos)),
        buttons=MouseButton.LEFT,
    )
    interaction.handle_event(press_event, cam)
    # Simulate mouse move (orbit) of one horizontal pixel and one vertical pixel
    move_pos = (click_pos[0] + 1, click_pos[1] + 1)
    move_event = MouseMoveEvent(
        canvas_pos=move_pos,
        world_ray=_validate_ray(canvas.to_world(move_pos)),
        buttons=MouseButton.LEFT,
    )
    interaction.handle_event(move_event, cam)
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


def test_orbit_mouse_right() -> None:
    """Tests right-click panning"""
    # Camera is along the x axis, looking in the negative x direction at the center
    interaction = snx.Orbit(center=(0, 0, 0))
    cam = snx.Camera(interactive=True, controller=interaction)
    # Add cam to the canvas
    view = snx.View(camera=cam)
    canvas = snx.Canvas(views=[view])
    # Position the camera along the x-axis, looking in the negative x direction at the
    # center
    cam.transform = snx.Transform().rotated(90, (0, 1, 0)).translated((10, 0, 0))
    _, _, w, h = canvas.rect_for(view)
    ray = canvas.to_world((w / 2, h / 2))
    assert ray is not None
    np.testing.assert_allclose(ray.origin, (10, 0, 0), atol=1e-7)
    np.testing.assert_allclose(ray.direction, (-1, 0, 0), atol=1e-7)
    tform_before = cam.transform
    center_before = np.array(interaction.center)

    # Simulate right mouse press
    click_pos = (w / 2, h / 2)
    world_ray_before = canvas.to_world(click_pos)
    assert world_ray_before is not None
    press_event = MousePressEvent(
        canvas_pos=click_pos,
        world_ray=world_ray_before,
        buttons=MouseButton.RIGHT,
    )
    interaction.handle_event(press_event, cam)
    # Simulate right mouse move (pan)
    click_pos = (click_pos[0], click_pos[1] + int(h) // 2)
    world_ray_after = canvas.to_world(click_pos)
    assert world_ray_after is not None
    move_event = MouseMoveEvent(
        canvas_pos=click_pos,
        world_ray=world_ray_after,
        buttons=MouseButton.RIGHT,
    )
    interaction.handle_event(move_event, cam)
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
    np.testing.assert_allclose(interaction.center, desired_center)


def test_orbit_scroll() -> None:
    """Tests zooming via mouse wheel events on Orbit."""
    center = (0.0, 0.0, 0.0)
    interaction = snx.Orbit(center=center)
    starting_dist = 10
    cam = snx.Camera(
        interactive=True,
        transform=snx.Transform().translated((0, 0, starting_dist)),
        controller=interaction,
    )
    tform_before = cam.transform
    # Simulate wheel event
    delta = 120  # one wheel click
    wheel_event = WheelEvent(
        canvas_pos=(0, 0),
        world_ray=Ray((0, 0, 0), (0, 0, -1), source=MagicMock(spec=snx.View)),
        buttons=MouseButton.NONE,
        angle_delta=(0, delta),
    )
    interaction.handle_event(wheel_event, cam)
    # The camera should have moved closer to center
    zoom = interaction._zoom_factor(delta)
    desired_tform = snx.Transform().translated((0, 0, starting_dist / zoom))
    np.testing.assert_allclose(cam.transform, desired_tform)

    # Simulate wheel event in other direction
    delta = -120  # one wheel click in the other direction
    wheel_event = WheelEvent(
        canvas_pos=(0, 0),
        world_ray=Ray((0, 0, 0), (0, 0, -1), source=MagicMock(spec=snx.View)),
        buttons=MouseButton.NONE,
        angle_delta=(0, delta),
    )
    interaction.handle_event(wheel_event, cam)
    # The camera should have moved back to the starting point
    zoom = interaction._zoom_factor(delta)
    desired_tform = snx.Transform().translated((0, 0, starting_dist))
    np.testing.assert_allclose(cam.transform, tform_before)


def test_orbit_zoom() -> None:
    """Tests zooming via the public zoom() API on Orbit."""
    center = (0.0, 0.0, 0.0)
    interaction = snx.Orbit(center=center)
    starting_dist = 10
    cam = snx.Camera(
        interactive=True,
        transform=snx.Transform().translated((0, 0, starting_dist)),
        controller=interaction,
    )
    factor = 0.5  # zoom out slightly
    interaction.zoom(cam, factor)
    # Camera should have moved along the camera-to-center axis by (1 - factor)
    # So now it should be at `(1 + (1 - factor)) = 2 - factor` along the z axis
    desired_tform = snx.Transform().translated((0, 0, 20))
    np.testing.assert_allclose(cam.transform, desired_tform)

    factor = 2  # zoom in slightly
    interaction.zoom(cam, factor)
    desired_tform = snx.Transform().translated((0, 0, 10))
    np.testing.assert_allclose(cam.transform, desired_tform)


def test_panzoom_serialization() -> None:
    cam = snx.Camera(
        controller=snx.PanZoom(),
        interactive=True,
        transform=snx.Transform().translated((10, 20, 30)).scaled((2, 2, 2)),
    )
    json = cam.model_dump_json()
    cam2 = snx.Camera.model_validate_json(json)
    assert isinstance(cam2.controller, snx.PanZoom)


def test_orbit_serialization() -> None:
    center = (5, 5, 10)
    polar_axis = (1, 0, 0)
    cam = snx.Camera(
        controller=snx.Orbit(center=center, polar_axis=polar_axis),
        interactive=True,
        transform=snx.Transform().translated((10, 20, 30)).scaled((2, 2, 2)),
    )
    json = cam.model_dump_json()
    cam2 = snx.Camera.model_validate_json(json)
    assert isinstance(cam2.controller, snx.Orbit)
    assert cam2.controller.center == center
    assert cam2.controller.polar_axis == polar_axis
