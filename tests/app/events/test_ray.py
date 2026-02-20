from unittest.mock import MagicMock

import numpy as np

import scenex as snx
from scenex.app.events import Ray


def test_intersections() -> None:
    image = snx.Image(
        data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
        interactive=True,
        visible=True,
    )
    scene = snx.Scene(children=[image], interactive=True, visible=True)
    ray = Ray(origin=(50, 50, 1), direction=(0, 0, -1), source=MagicMock(spec=snx.View))

    # Test intersections where the image is visible
    intersections = ray.intersections(image)
    assert len(intersections) == 1
    assert intersections[0][0] is image
    intersections = ray.intersections(scene)
    assert len(intersections) == 1
    assert intersections[0][0] is image

    # Test intersections where the image is not visible
    image.visible = False
    intersections = ray.intersections(image)
    assert len(intersections) == 0
    intersections = ray.intersections(scene)
    assert len(intersections) == 0

    # Test intersections where the image is visible, but not interactive
    image.visible = True
    image.interactive = False
    intersections = ray.intersections(image)
    assert len(intersections) == 1
    assert intersections[0][0] is image
    intersections = ray.intersections(scene)
    assert len(intersections) == 1
    assert intersections[0][0] is image
