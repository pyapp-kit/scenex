import numpy as np

import scenex as snx
from scenex.model._transform import Transform


def test_bounding_box() -> None:
    # An empty scene should have an "empty" bounding box
    empty_scene = snx.Scene()
    # TODO: Is the bounding box of an empty node really zeroes? Need to think more about
    # that.
    exp_bounding_box = np.zeros((2, 3))
    assert np.array_equal(exp_bounding_box, empty_scene.bounding_box)

    # With children, the bounding box should be defined by the scene
    points_scene = snx.Scene(
        children=[snx.Points(coords=np.asarray([[0, 100, 0], [100, 0, 1]]))]
    )
    exp_bounding_box = np.asarray(((0, 0, 0), (100, 100, 1)))
    assert np.array_equal(exp_bounding_box, points_scene.bounding_box)

    # The bounding box should move as their points move.
    points_scene = snx.Scene(
        children=[
            snx.Points(
                coords=np.asarray([[0, 100, 0], [100, 0, 1]]),
                transform=Transform().translated((1, 1, 1)),
            )
        ]
    )
    exp_bounding_box = np.asarray(((1, 1, 1), (101, 101, 2)))
    assert np.array_equal(exp_bounding_box, points_scene.bounding_box)

    # The bounding box should encapsulate all children
    points_scene = snx.Scene(
        children=[
            snx.Points(
                coords=np.asarray(
                    [
                        [0, 100, 0],
                    ]
                ),
            ),
            snx.Points(
                coords=np.asarray(
                    [
                        [100, 0, 1],
                    ]
                ),
            ),
        ]
    )
    exp_bounding_box = np.asarray(((0, 0, 0), (100, 100, 1)))
    assert np.array_equal(exp_bounding_box, points_scene.bounding_box)
