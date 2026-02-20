import numpy as np

import scenex as snx
from scenex.model._transform import Transform


def test_bounding_box() -> None:
    # An empty scene should have an "empty" bounding box
    empty_scene = snx.Scene()
    assert empty_scene.bounding_box is None

    # With children, the bounding box should be defined by the scene
    points_scene = snx.Scene(
        children=[snx.Points(vertices=np.asarray([[0, 100, 0], [100, 0, 1]]))]
    )
    exp_bounding_box = np.asarray(((0, 0, 0), (100, 100, 1)))
    act_bouning_box = points_scene.bounding_box
    assert act_bouning_box is not None
    assert np.array_equal(exp_bounding_box, act_bouning_box)

    # The bounding box should move as their points move.
    points_scene = snx.Scene(
        children=[
            snx.Points(
                vertices=np.asarray([[0, 100, 0], [100, 0, 1]]),
                transform=Transform().translated((1, 1, 1)),
            )
        ]
    )
    exp_bounding_box = np.asarray(((1, 1, 1), (101, 101, 2)))
    act_bouning_box = points_scene.bounding_box
    assert act_bouning_box is not None
    assert np.array_equal(exp_bounding_box, act_bouning_box)

    # The bounding box should encapsulate all children
    points_scene = snx.Scene(
        children=[
            snx.Points(
                vertices=np.asarray(
                    [
                        [0, 100, 0],
                    ]
                ),
            ),
            snx.Points(
                vertices=np.asarray(
                    [
                        [100, 0, 1],
                    ]
                ),
            ),
        ]
    )
    exp_bounding_box = np.asarray(((0, 0, 0), (100, 100, 1)))
    act_bouning_box = points_scene.bounding_box
    assert act_bouning_box is not None
    assert np.array_equal(exp_bounding_box, act_bouning_box)
