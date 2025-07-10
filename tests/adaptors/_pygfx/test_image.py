from __future__ import annotations

import numpy as np
import pytest

import scenex as snx
import scenex.adaptors._pygfx as adaptors
from scenex.model._transform import Transform


@pytest.fixture
def image() -> snx.Image:
    return snx.Image(
        data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
    )


@pytest.fixture
def adaptor(image: snx.Image) -> adaptors.Image:
    adaptor = image._get_adaptor(create=True)
    assert isinstance(adaptor, adaptors.Image)
    return adaptor


def test_transform(image: snx.Image, adaptor: adaptors.Image) -> None:
    # FIXME: Ideally we could just use get_bounding_box(), which purports
    # to return a bounding box in parent space, however the local transform
    # of the Image is not used

    # No Transform
    image.transform = Transform()
    exp_bounds = np.asarray([[-0.5, -0.5, 0], [99.5, 99.5, 0]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Just Translation
    image.transform = Transform().translated((-10, -10))
    exp_bounds = np.asarray([[-10.5, -10.5, 0], [89.5, 89.5, 0]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Just Scaling
    image.transform = Transform().scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-0.25, -0.25, 0], [49.75, 49.75, 0]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Scaling then Translation
    image.transform = Transform().scaled((0.5, 0.5, 0.5)).translated((-10, -10))
    exp_bounds = np.asarray([[-10.25, -10.25, 0], [39.75, 39.75, 0]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Translation then Scaling
    image.transform = Transform().translated((-10, -10)).scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-5.25, -5.25, 0], [44.75, 44.75, 0]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)
