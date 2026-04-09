from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import scenex as snx
import scenex.adaptors._vispy as adaptors
from scenex.adaptors._auto import get_adaptor_registry
from scenex.adaptors._vispy._image import _get_max_texture_sizes
from scenex.model._transform import Transform

if TYPE_CHECKING:
    from vispy.visuals import ImageVisual


@pytest.fixture
def image() -> snx.Image:
    # Width (x-axis) is 80 and Height (y-axis) is 100
    return snx.Image(
        data=np.random.randint(0, 255, (100, 80), dtype=np.uint8),
    )


@pytest.fixture
def adaptor(image: snx.Image) -> adaptors.Image:
    adaptor = get_adaptor_registry().get_adaptor(image, create=True)
    assert isinstance(adaptor, adaptors.Image)
    return adaptor


def test_transform(image: snx.Image, adaptor: adaptors.Image) -> None:
    # No Transform
    image.transform = Transform()
    exp_bounds = np.asarray([[-0.5, -0.5, 0], [79.5, 99.5, 0]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Just Translation
    image.transform = Transform().translated((-10, -10))
    exp_bounds = np.asarray([[-10.5, -10.5, 0], [69.5, 89.5, 0]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Just Scaling
    image.transform = Transform().scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-0.25, -0.25, 0], [39.75, 49.75, 0]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Scaling then Translation
    image.transform = Transform().scaled((0.5, 0.5, 0.5)).translated((-10, -10))
    exp_bounds = np.asarray([[-10.25, -10.25, 0], [29.75, 39.75, 0]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Translation then Scaling
    image.transform = Transform().translated((-10, -10)).scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-5.25, -5.25, 0], [34.75, 44.75, 0]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)


def test_oversized_texture() -> None:
    """Demonstrates that textures exceeding GPU dimension limits are downsampled.

    When image data exceeds the GPU's max texture dimension, vispy will silently send
    smaller data (TODO: Better explanation - this particular size shows a single pixel)
    The adaptor should detect oversized data and transparently downsample it (e.g. via a
    strided view) before uploading, avoiding this erroneous behavior.
    """
    max_dim = _get_max_texture_sizes()[0]  # 2D limit
    if max_dim is None:
        pytest.skip("GPU does not report a max texture size, cannot test downsampling")

    # Create an image that exceeds the GPU limits
    oversized_shape = (1, max_dim + 1)
    image = snx.Image(data=np.zeros(oversized_shape, dtype=np.uint8))
    # Create an adaptor for it
    adaptor = image._get_adaptors(create=True)[0]
    assert isinstance(adaptor, adaptors.Image)
    # And check the texture was downsampled to fit within the GPU limits
    assert adaptor._vispy_node._data.shape == (  # pyright: ignore
        oversized_shape[0],
        (oversized_shape[1] + 1) // 2,
    )
    # But ensure that the adaptor's transform is still correctly compensating for the
    # downsampling, so the image appears at the correct size
    bb = _bounds(adaptor._vispy_node)
    np.testing.assert_almost_equal(bb, image.bounding_box)


@pytest.mark.parametrize(
    ("src_dtype", "expected_dtype"),
    [
        (np.float64, np.float32),
        (np.int32, np.uint16),
        (np.int64, np.uint16),
        (np.uint32, np.uint16),
        (np.uint64, np.uint16),
    ],
)
def test_downcasting(src_dtype: np.dtype, expected_dtype: np.dtype) -> None:
    """Vispy does not allow 64-bit data. This test ensures 64-bit data is downcasted."""
    rng = np.random.default_rng()
    if np.issubdtype(src_dtype, np.floating):
        data = rng.random((100, 100), dtype=src_dtype)
    else:
        data = rng.integers(0, 255, (100, 100), dtype=src_dtype)
    image = snx.Image(data=data)
    adaptor = get_adaptor_registry().get_adaptor(image, create=True)
    assert isinstance(adaptor, adaptors.Image)
    assert adaptor._vispy_node._data.dtype == expected_dtype  # pyright: ignore


def _bounds(node: ImageVisual) -> np.ndarray:
    bounds: np.ndarray = np.ndarray((2, 3), dtype=np.float32)
    # Get the bounds of the raw object...
    for i in range(3):
        b = node.bounds(i)
        bounds[:, i] = min(b), max(b)
    # ...and transform them
    return cast("np.ndarray", node.transform.map(bounds)[:, :3])
