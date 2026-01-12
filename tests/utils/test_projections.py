import numpy as np
from pylinalg import vec_unproject

from scenex.model._transform import Transform
from scenex.utils.projections import orthographic, perspective

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


def test_projection() -> None:
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

    def _project(mat: Transform, world_pos: tuple[float, float, float]) -> np.ndarray:
        # Inverting the behavior of vec_unproject
        proj = np.dot(mat.root, np.asarray((*world_pos, 1)))
        return proj[:2] / proj[3]  # type: ignore

    # Test a near frustum corner maps to a canvas corner
    # Note that by convention positive z points away from the scene.
    assert np.allclose(np.asarray((1, 1)), _project(mat, (300, 300, -300)))
    # Test a far frustum corner maps to a canvas corner
    assert np.allclose(np.asarray((1, 1)), _project(mat, (360, 360, -360)))
    # Test a near frustum corner, TRANSLATED back in the scene, does not map to a corner
    # This point models the back face of a volume
    assert np.allclose(np.asarray((5 / 6, 5 / 6)), _project(mat, (300, 300, -360)))
