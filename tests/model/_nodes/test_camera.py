import numpy as np

import scenex as snx


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
