from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pygfx
import pytest

import scenex as snx
import scenex.adaptors._pygfx as adaptors
from scenex.adaptors._auto import get_adaptor_registry
from scenex.model import BlendMode, Transform

# In various tests, we want to make sure that Textures aren't being needlessly recreated
# We patch the Texture class to count how many times it's called, but we need to be
# careful to patch the correct path - i.e. the path where the VolumeAdaptor looks up the
# Texture class, not necessarily the path where it's defined.
_TEXTURE_PATH = "scenex.adaptors._pygfx._volume.pygfx.Texture"


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
    bb = adaptor._pygfx_node.get_world_bounding_box()
    np.testing.assert_array_almost_equal(exp_bounds, bb)

    # Just Translation
    volume.transform = Transform().translated((-10, -10, -10))
    exp_bounds = np.asarray([[-10.5, -10.5, -10.5], [9.5, 19.5, 29.5]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    np.testing.assert_array_almost_equal(exp_bounds, bb)

    # Just Scaling
    volume.transform = Transform().scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-0.25, -0.25, -0.25], [9.75, 14.75, 19.75]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    np.testing.assert_array_almost_equal(exp_bounds, bb)

    # Scaling then Translation
    volume.transform = Transform().scaled((0.5, 0.5, 0.5)).translated((-10, -10, -10))
    exp_bounds = np.asarray([[-10.25, -10.25, -10.25], [-0.25, 4.75, 9.75]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    np.testing.assert_array_almost_equal(exp_bounds, bb)

    # Translation then Scaling
    volume.transform = Transform().translated((-10, -10, -10)).scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-5.25, -5.25, -5.25], [4.75, 9.75, 14.75]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    np.testing.assert_array_almost_equal(exp_bounds, bb)


@pytest.mark.skipif(
    pygfx.version_info < (0, 13, 0), reason="Requires pygfx 0.13.0 or higher"
)
def test_blending(volume: snx.Volume, adaptor: adaptors.Volume) -> None:
    volume.blending = BlendMode.ADDITIVE
    assert adaptor._material.alpha_mode == "add"

    volume.blending = BlendMode.ALPHA
    assert adaptor._material.alpha_mode == "auto"

    volume.blending = BlendMode.OPAQUE
    assert adaptor._material.alpha_mode == "solid"


@pytest.mark.parametrize(
    ("src_dtype", "expected_dtype"),
    [
        (np.float64, np.float32),
        (np.int64, np.int32),
        (np.uint64, np.uint32),
    ],
)
def test_downcasting(
    src_dtype: np.dtype, expected_dtype: np.dtype, caplog: pytest.LogCaptureFixture
) -> None:
    """Pygfx does not allow 64-bit data. This test ensures 64-bit data is downcasted."""
    rng = np.random.default_rng()
    if np.issubdtype(src_dtype, np.floating):
        data = rng.random((3, 100, 100), dtype=src_dtype)
    else:
        data = rng.integers(0, 255, (3, 100, 100), dtype=src_dtype)
    volume = snx.Volume(data=data)
    adaptor = get_adaptor_registry().get_adaptor(volume, create=True)
    assert isinstance(adaptor, adaptors.Volume)
    assert adaptor._texture.data.dtype == expected_dtype  # pyright: ignore


def test_oversized_texture() -> None:
    """Demonstrates that textures exceeding GPU dimension limits are downsampled."""
    import wgpu

    # Determine the GPU's max texture dimension
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    max_dim = device.limits["max-texture-dimension-3d"]

    # Create an image that exceeds the GPU limits
    oversized_shape = (1, 1, max_dim + 1)
    volume = snx.Volume(data=np.zeros(oversized_shape, dtype=np.uint8))
    # Create an adaptor for it
    adaptor = get_adaptor_registry().get_adaptor(volume, create=True)
    assert isinstance(adaptor, adaptors.Volume)
    # And check the texture was downsampled to fit within the GPU limits
    assert adaptor._texture.data.shape == (  # pyright: ignore
        oversized_shape[0],
        oversized_shape[1],
        (oversized_shape[2] + 1) // 2,
    )
    # But ensure that the adaptor's transform is still correctly compensating for the
    # downsampling, so the image appears at the correct size
    np.testing.assert_almost_equal(
        adaptor._pygfx_node.get_world_bounding_box(), volume.bounding_box
    )


def test_texture_reuse(volume: snx.Volume, adaptor: adaptors.Volume) -> None:
    """Updating data with the same shape and dtype reuses the existing texture."""
    new_data = np.full(volume.data.shape, 42, dtype=np.uint8)
    with patch(_TEXTURE_PATH, wraps=pygfx.Texture) as mock_texture:
        volume.data = new_data
    assert mock_texture.call_count == 0
    np.testing.assert_array_equal(adaptor._texture.data, new_data)  # pyright: ignore


def test_texture_recreated_on_shape_change(
    volume: snx.Volume, adaptor: adaptors.Volume
) -> None:
    """Updating data with a different shape creates a new texture."""
    existing_shape = adaptor._texture.data.shape  # pyright: ignore
    new_shape = tuple(s + 10 for s in existing_shape)
    with patch(_TEXTURE_PATH, wraps=pygfx.Texture) as mock_texture:
        volume.data = np.zeros(new_shape, dtype=np.uint8)
    assert mock_texture.call_count == 1


def test_texture_recreated_on_dtype_change(
    volume: snx.Volume, adaptor: adaptors.Volume
) -> None:
    """Updating data with a different dtype creates a new texture."""
    assert adaptor._texture.data.dtype != np.float32  # pyright: ignore
    with patch(_TEXTURE_PATH, wraps=pygfx.Texture) as mock_texture:
        volume.data = np.zeros((10, 10, 10), dtype=np.float32)
    assert mock_texture.call_count == 1


def test_texture_buffer_not_source_array(
    volume: snx.Volume, adaptor: adaptors.Volume
) -> None:
    """Reusing pygfx Textures opens the door to mutation - test it doesn't happen."""
    old_data_actual = volume.data
    assert isinstance(old_data_actual, np.ndarray)
    old_data_expected = old_data_actual.copy()
    volume.data = old_data_actual + 1
    assert np.array_equal(old_data_actual, old_data_expected)
