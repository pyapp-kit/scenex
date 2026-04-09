from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import scenex as snx
import scenex.adaptors._vispy as adaptors
from scenex.adaptors._auto import get_adaptor_registry
from scenex.adaptors._vispy._image import _get_max_texture_sizes
from scenex.model import BlendMode, Transform

if TYPE_CHECKING:
    from vispy.visuals import VolumeVisual


@pytest.fixture
def volume() -> snx.Volume:
    return snx.Volume(
        data=np.random.randint(0, 255, (40, 30, 20), dtype=np.uint8),
    )


@pytest.fixture
def adaptor(volume: snx.Volume) -> adaptors.Volume:
    adaptor = get_adaptor_registry().get_adaptor(volume, create=True)
    assert isinstance(adaptor, adaptors.Volume)
    return adaptor


def test_transform(volume: snx.Volume, adaptor: adaptors.Volume) -> None:
    # No Transform
    volume.transform = Transform()
    exp_bounds = np.asarray([[-0.5, -0.5, -0.5], [19.5, 29.5, 39.5]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Just Translation
    volume.transform = Transform().translated((-10, -10, -10))
    exp_bounds = np.asarray([[-10.5, -10.5, -10.5], [9.5, 19.5, 29.5]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Just Scaling
    volume.transform = Transform().scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-0.25, -0.25, -0.25], [9.75, 14.75, 19.75]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Scaling then Translation
    volume.transform = Transform().scaled((0.5, 0.5, 0.5)).translated((-10, -10, -10))
    exp_bounds = np.asarray([[-10.25, -10.25, -10.25], [-0.25, 4.75, 9.75]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Translation then Scaling
    volume.transform = Transform().translated((-10, -10, -10)).scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-5.25, -5.25, -5.25], [4.75, 9.75, 14.75]])
    bb = _bounds(adaptor._vispy_node)
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)


def test_blending(volume: snx.Volume, adaptor: adaptors.Volume) -> None:
    # Test blending modes
    from unittest.mock import MagicMock

    # Note that we can't just get the gl state, so we should assert that the correct
    # settings were set.
    adaptor._vispy_node.set_gl_state = MagicMock()

    volume.blending = BlendMode.ADDITIVE
    adaptor._vispy_node.set_gl_state.assert_called_with("additive")

    volume.blending = BlendMode.ALPHA
    adaptor._vispy_node.set_gl_state.assert_called_with("translucent")

    volume.blending = BlendMode.OPAQUE
    adaptor._vispy_node.set_gl_state.assert_called_with(None, blend=False)


@pytest.mark.parametrize(
    ("src_dtype", "expected_dtype"),
    [
        (np.float64, np.float32),
        (np.int32, np.float32),
        (np.int64, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
    ],
)
def test_casting(
    src_dtype: np.dtype, expected_dtype: np.dtype, caplog: pytest.LogCaptureFixture
) -> None:
    """Vispy does not allow several dtypes. This test ensures those are casted."""
    rng = np.random.default_rng()
    if np.issubdtype(src_dtype, np.floating):
        data = rng.random((3, 100, 100), dtype=src_dtype)
    else:
        data = rng.integers(0, 255, (3, 100, 100), dtype=src_dtype)
    volume = snx.Volume(data=data)
    adaptor = get_adaptor_registry().get_adaptor(volume, create=True)
    assert isinstance(adaptor, adaptors.Volume)
    assert adaptor._vispy_node._texture._data_dtype == expected_dtype  # pyright: ignore


def test_oversized_texture() -> None:
    """Demonstrates that textures exceeding GPU dimension limits are downsampled."""
    max_dim = _get_max_texture_sizes()[1]  # 3D limit
    if max_dim is None:
        pytest.skip("GPU does not report a max texture size, cannot test downsampling")

    # Create an image that exceeds the GPU limits in the X dimension
    # NOTE volumes are ZYX order
    oversized_shape = (1, 2, max_dim + 1)
    volume = snx.Volume(data=np.zeros(oversized_shape, dtype=np.uint8))
    # Create an adaptor for it
    adaptor = get_adaptor_registry().get_adaptor(volume, create=True)
    assert isinstance(adaptor, adaptors.Volume)
    # And check the texture was downsampled to fit within the GPU limits
    assert adaptor._vispy_node._texture.shape == (  # pyright: ignore
        oversized_shape[0],
        oversized_shape[1],
        (oversized_shape[2] + 1) // 2,
        1,
    )
    # But ensure that the adaptor's transform is still correctly compensating for the
    # downsampling, so the image appears at the correct size
    bb = _bounds(adaptor._vispy_node)
    np.testing.assert_almost_equal(bb, volume.bounding_box)


def _bounds(node: VolumeVisual) -> np.ndarray:
    bounds: np.ndarray = np.ndarray((2, 3), dtype=np.float32)
    # Get the bounds of the raw object...
    for i in range(3):
        b = node.bounds(i)
        bounds[:, i] = min(b), max(b)
    # ...and transform them
    return cast("np.ndarray", node.transform.map(bounds)[:, :3])
