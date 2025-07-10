from __future__ import annotations

import numpy as np
import pytest

import scenex as snx
import scenex.adaptors._pygfx as adaptors
from scenex.model._transform import Transform


@pytest.fixture
def volume() -> snx.Volume:
    return snx.Volume(
        data=np.random.randint(0, 255, (100, 100, 100), dtype=np.uint8),
    )


@pytest.fixture
def adaptor(volume: snx.Volume) -> adaptors.Volume:
    adaptor = volume._get_adaptor(create=True)
    assert isinstance(adaptor, adaptors.Volume)
    return adaptor


def test_transform(volume: snx.Volume, adaptor: adaptors.Volume) -> None:
    # No Transform
    volume.transform = Transform()
    exp_bounds = np.asarray([[-0.5, -0.5, -0.5], [99.5, 99.5, 99.5]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Just Translation
    volume.transform = Transform().translated((-10, -10, -10))
    exp_bounds = np.asarray([[-10.5, -10.5, -10.5], [89.5, 89.5, 89.5]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Just Scaling
    volume.transform = Transform().scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-0.25, -0.25, -0.25], [49.75, 49.75, 49.75]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Scaling then Translation
    volume.transform = Transform().scaled((0.5, 0.5, 0.5)).translated((-10, -10, -10))
    exp_bounds = np.asarray([[-10.25, -10.25, -10.25], [39.75, 39.75, 39.75]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Translation then Scaling
    volume.transform = Transform().translated((-10, -10, -10)).scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-5.25, -5.25, -5.25], [44.75, 44.75, 44.75]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)
