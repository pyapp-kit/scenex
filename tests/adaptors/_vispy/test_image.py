from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import scenex as snx
import scenex.adaptors._vispy as adaptors
from scenex.adaptors._auto import get_adaptor_registry
from scenex.model._transform import Transform

if TYPE_CHECKING:
    from vispy.visuals import ImageVisual


@pytest.fixture
def image() -> snx.Image:
    return snx.Image(
        data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
    )


@pytest.fixture
def adaptor(image: snx.Image) -> adaptors.Image:
    adaptor = get_adaptor_registry().get_adaptor(image, create=True)
    assert isinstance(adaptor, adaptors.Image)
    return adaptor


def test_transform(image: snx.Image, adaptor: adaptors.Image) -> None:
    # No Transform
    image.transform = Transform()
    exp_bounds = np.asarray([[-0.5, -0.5, 0], [99.5, 99.5, 0]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Just Translation
    image.transform = Transform().translated((-10, -10))
    exp_bounds = np.asarray([[-10.5, -10.5, 0], [89.5, 89.5, 0]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Just Scaling
    image.transform = Transform().scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-0.25, -0.25, 0], [49.75, 49.75, 0]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Scaling then Translation
    image.transform = Transform().scaled((0.5, 0.5, 0.5)).translated((-10, -10))
    exp_bounds = np.asarray([[-10.25, -10.25, 0], [39.75, 39.75, 0]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Translation then Scaling
    image.transform = Transform().translated((-10, -10)).scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-5.25, -5.25, 0], [44.75, 44.75, 0]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)


def _bounds(node: ImageVisual) -> np.ndarray:
    bounds: np.ndarray = np.ndarray((2, 3), dtype=np.float32)
    # Get the bounds of the raw object...
    for i in range(3):
        b = node.bounds(i)
        bounds[:, i] = min(b), max(b)
    # ...and transform them
    return cast("np.ndarray", node.transform.map(bounds)[:, :3])
