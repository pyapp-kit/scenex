import numpy as np
from pylinalg import vec_unproject

import scenex as snx
from scenex.model._transform import Transform
from scenex.utils.projections import orthographic, perspective, zoom_to_fit

CORNERS = np.asarray(
    [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ]
)


def test_orthographic() -> None:
    """Basic tests for the orthographic projection matrix constructor"""
    # By default, the return should unproject NDCs to a depth-inverted unit cube
    exp_mat = np.asarray(
        [
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, -2, 0],
            [0, 0, 0, 1],
        ]
    )
    act_mat = orthographic()
    assert np.array_equal(exp_mat, act_mat)
    exp_corners = np.asarray(
        [
            [-0.5, -0.5, 0],
            [-0.5, 0.5, 0],
            [0.5, -0.5, 0],
            [0.5, 0.5, 0],
        ]
    )
    assert np.array_equal(exp_corners, vec_unproject(CORNERS, act_mat))

    # Scales inversely w.r.t. width, height, depth
    exp_mat = np.asarray(
        [
            [2 / 3, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, -2 / 5, 0],
            [0, 0, 0, 1],
        ]
    )
    act_mat = orthographic(3, 4, 5)  # Cube with width 3, height 4, depth 5 in view
    exp_corners = np.asarray(
        [
            [-1.5, -2, 0],
            [-1.5, 2, 0],
            [1.5, -2, 0],
            [1.5, 2, 0],
        ]
    )
    assert np.array_equal(exp_corners, vec_unproject(CORNERS, act_mat))


def test_perspective() -> None:
    """Basic testing of the perspective matrix"""
    fov = 90
    depth_to_near = 300
    depth_to_far = 1e6  # Just need something really big

    mat = perspective(fov, depth_to_near, depth_to_far)

    exp_mat = np.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, -300],
            [0, 0, -1, 0],
        ]
    )

    # Note the z-offset is like 300.09. Might be rounding errors?
    assert np.allclose(exp_mat, mat, rtol=1e-1)

    # Test a near frustum corner maps to a canvas corner
    # Note that by convention positive z points away from the scene.
    assert np.allclose(np.asarray((1, 1)), _project(mat, (300, 300, -300)))
    # Test a far frustum corner maps to a canvas corner
    assert np.allclose(np.asarray((1, 1)), _project(mat, (360, 360, -360)))
    # Test a near frustum corner, TRANSLATED back in the scene, does not map to a corner
    # This point models the back face of a volume
    assert np.allclose(np.asarray((5 / 6, 5 / 6)), _project(mat, (300, 300, -360)))


def test_zoom_to_fit_orthographic() -> None:
    view = snx.View(
        scene=snx.Scene(
            children=[snx.Points(coords=np.asarray([[0, 100, 0], [100, 0, 1]]))]
        )
    )

    zoom_to_fit(view, type="orthographic")
    # Assert the camera is moved to the center of the scene
    assert view.camera.transform == snx.Transform().translated((50, 50, 0.5))
    # Projection that maps world space to canvas coordinates
    tform = view.camera.transform.inv() @ view.camera.projection
    # Assert the camera projects [0, 0, 0] to NDC coordinates [-1, -1]
    assert np.array_equal((-1, -1), tform.map((0, 0, 0))[:2])
    # ...and [100, 100, 0] to NDC coordinates [1, 1]
    assert np.array_equal((1, 1), tform.map((100, 100, 0))[:2])

    zoom_factor = 0.9
    zoom_to_fit(view, type="orthographic", zoom_factor=zoom_factor)
    # Assert the camera is still at the center of the scene
    assert view.camera.transform == snx.Transform().translated((50, 50, 0.5))
    # Projection that maps world space to canvas coordinates
    tform = view.camera.transform.inv() @ view.camera.projection
    # Assert the camera projects [0, 0, 0] to NDC coordinates [-0.9, -0.9]
    assert np.allclose(
        (-zoom_factor, -zoom_factor), tform.map((0, 0, 0))[:2], rtol=1e-10
    )
    # ...and [100, 100, 0] to NDC coordinates [0.9, 0.9]
    assert np.allclose(
        (zoom_factor, zoom_factor), tform.map((100, 100, 0))[:2], rtol=1e-10
    )


def test_zoom_to_fit_perspective() -> None:
    view = snx.View(
        scene=snx.Scene(
            children=[snx.Points(coords=np.asarray([[0, 100, 1], [100, 0, 0]]))]
        )
    )
    zoom_to_fit(view, type="perspective")

    # Assert the camera is moved to the center of the scene
    # Depth isn't particularly important to test here.
    assert np.array_equal(
        view.camera.transform,
        Transform().translated((50, 50, view.camera.transform.root[3, 2])),
    )
    # Projection that maps world space to canvas coordinates
    tform = view.camera.projection @ view.camera.transform.inv().T
    # Assert the camera projects [0, 0, 0] to NDC coordinates [-1, -1]
    assert np.array_equal((-1, -1), _project(tform, (0, 0, 1))[:2])
    # ...and [100, 100, 0] to NDC coordinates [1, 1]
    assert np.array_equal((1, 1), _project(tform, (100, 100, 1))[:2])
    # And assert the entire scene is within the canvas:
    assert np.all(_project(tform, (0, 0, 0)) > _project(tform, (0, 0, 1)))
    assert np.all(_project(tform, (100, 100, 0)) < _project(tform, (100, 100, 1)))

    zoom_factor = 0.9
    zoom_to_fit(view, type="perspective", zoom_factor=zoom_factor)
    # Assert the camera is still at the center of the scene
    assert np.array_equal(
        view.camera.transform,
        Transform().translated((50, 50, view.camera.transform.root[3, 2])),
    )
    # Projection that maps world space to canvas coordinates
    tform = view.camera.projection @ view.camera.transform.inv().T
    # Assert the camera projects [0, 0, 0] to NDC coordinates [-0.9, -0.9]
    assert np.allclose(
        (-zoom_factor, -zoom_factor), _project(tform, (0, 0, 1))[:2], rtol=1e-10
    )
    # ...and [100, 100, 0] to NDC coordinates [0.9, 0.9]
    assert np.allclose(
        (zoom_factor, zoom_factor), _project(tform, (100, 100, 1))[:2], rtol=1e-10
    )
    # And assert the entire scene is within the canvas:
    assert np.all(_project(tform, (0, 0, 0)) > _project(tform, (0, 0, 1)))
    assert np.all(_project(tform, (100, 100, 0)) < _project(tform, (100, 100, 1)))


def _project(mat: snx.Transform, world_pos: tuple[float, float, float]) -> np.ndarray:
    # Inverting the behavior of vec_unproject
    proj = np.dot(mat.root, np.asarray((*world_pos, 1)))
    return proj[:2] / proj[3]  # type: ignore
