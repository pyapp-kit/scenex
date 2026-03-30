import math
from collections.abc import Generator

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
from scenex.utils import projections


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


@pytest.fixture
def ortho_view() -> Generator[snx.View, None, None]:
    # Create a camera showing min=(-50, 50), max=(50, -50)
    cam = snx.Camera(projection=projections.orthographic(width=100, height=100))
    # Put it in a view...
    view = snx.View(camera=cam)
    # ...on a canvas, so that it has a size and can convert to world coordinates
    canvas = snx.Canvas(views=[view], width=100, height=100)  # noqa: F841
    # Note that we yield to hold onto the canvas ref
    yield view


def test_panzoom_pan(ortho_view: snx.View) -> None:
    """Tests panning behavior of PanZoom."""
    interaction = ortho_view.camera.controller = snx.PanZoom()
    # Simulate mouse press at canvas (0, 0), world (-50, 50)
    press_event = MousePressEvent(
        pos=(0, 0),
        buttons=MouseButton.LEFT,
    )
    interaction.handle_event(press_event, ortho_view)
    # Simulate mouse move to canvas (5, 10)
    move_event = MouseMoveEvent(
        pos=(5, 10),
        buttons=MouseButton.LEFT,
    )
    interaction.handle_event(move_event, ortho_view)
    # The camera should have moved by (-5, 10) to keep (-50, 50) under the cursor
    # (under canvas (0, 0) should now be world (-45, 60))
    expected = snx.Transform().translated((-5, 10))
    np.testing.assert_allclose(ortho_view.camera.transform.root, expected.root)


def test_panzoom_zoom(ortho_view: snx.View) -> None:
    """Tests zooming behavior of PanZoom."""
    interaction = ortho_view.camera.controller = snx.PanZoom()
    # Simulate wheel event
    wheel_event = WheelEvent(
        pos=(0, 0),
        buttons=MouseButton.NONE,
        angle_delta=(0, 120),
    )
    before = ortho_view.camera.projection
    interaction.handle_event(wheel_event, ortho_view)
    # The projection should be scaled
    zoom = interaction._zoom_factor(wheel_event.angle_delta[1])
    expected = before.scaled((zoom, zoom, 1))
    np.testing.assert_allclose(ortho_view.camera.projection.root, expected.root)


def test_orbit_orbiting() -> None:
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
        pos=click_pos,
        buttons=MouseButton.LEFT,
    )
    interaction.handle_event(press_event, view)
    # Simulate mouse move (orbit) of one horizontal pixel and one vertical pixel
    move_pos = (click_pos[0] + 1, click_pos[1] + 1)
    move_event = MouseMoveEvent(
        pos=move_pos,
        buttons=MouseButton.LEFT,
    )
    interaction.handle_event(move_event, view)
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


def test_orbit_zoom() -> None:
    center = (0.0, 0.0, 0.0)
    interaction = snx.Orbit(center=center)
    cam = snx.Camera(
        interactive=True,
        transform=snx.Transform().translated((0, 0, 10)),
        controller=interaction,
    )
    # Add cam to the canvas
    view = snx.View(camera=cam)
    canvas = snx.Canvas(views=[view])  # noqa: F841
    tform_before = cam.transform
    # Simulate wheel event
    wheel_event = WheelEvent(
        pos=(0, 0),
        buttons=MouseButton.NONE,
        angle_delta=(0, 120),
    )
    interaction.handle_event(wheel_event, view)
    # The camera should have moved closer to center
    zoom = interaction._zoom_factor(120)
    desired_tform = snx.Transform().translated((0, 0, 10 * zoom))
    np.testing.assert_allclose(cam.transform, desired_tform)

    # Simulate wheel event in other direction
    wheel_event = WheelEvent(
        pos=(0, 0),
        buttons=MouseButton.NONE,
        angle_delta=(0, -120),
    )
    interaction.handle_event(wheel_event, view)
    # The camera should have moved back to the starting point
    zoom = interaction._zoom_factor(-120)
    desired_tform = snx.Transform().translated((0, 0, 10))
    np.testing.assert_allclose(cam.transform, tform_before)


def test_orbit_pan() -> None:
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
        pos=click_pos,
        buttons=MouseButton.RIGHT,
    )
    interaction.handle_event(press_event, view)
    # Simulate right mouse move (pan)
    click_pos = (click_pos[0], click_pos[1] + int(h) // 2)
    world_ray_after = canvas.to_world(click_pos)
    assert world_ray_after is not None
    move_event = MouseMoveEvent(
        pos=click_pos,
        buttons=MouseButton.RIGHT,
    )
    interaction.handle_event(move_event, view)
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
