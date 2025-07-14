import numpy as np
from pylinalg import vec_unproject

from scenex.utils.projections import orthographic

CORNERS = np.asarray(
    [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ]
)


def test_orthographic() -> None:
    """Smoke testing the orthographic matrix"""
    # By default, the return should unproject NDCs to a depth-inverted unit cube
    exp_mat = np.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )
    act_mat = orthographic()
    assert np.array_equal(exp_mat, act_mat)
    exp_corners = np.asarray(
        [
            [-1, -1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [1, 1, 0],
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
