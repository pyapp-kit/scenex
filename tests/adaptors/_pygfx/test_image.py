from __future__ import annotations

from unittest.mock import patch

import cmap
import numpy as np
import pygfx
import pytest

import scenex as snx
import scenex.adaptors._pygfx as adaptors
from scenex.adaptors import get_adaptor_registry
from scenex.model._transform import Transform

# In various tests, we want to make sure that Textures aren't being needlessly recreated
# We patch the Texture class to count how many times it's called, but we need to be
# careful to patch the correct path - i.e. the path where the ImageAdaptor looks up the
# Texture class, not necessarily the path where it's defined.
_TEXTURE_PATH = "scenex.adaptors._pygfx._image.pygfx.Texture"


@pytest.fixture
def image() -> snx.Image:
    return snx.Image(
        data=np.random.randint(0, 255, (200, 100), dtype=np.uint8),
        cmap=cmap.Colormap("viridis"),
    )


@pytest.fixture
def adaptor(image: snx.Image) -> adaptors.Image:
    adaptor = get_adaptor_registry().get_adaptor(image, create=True)
    assert isinstance(adaptor, adaptors.Image)
    return adaptor


def test_transform(image: snx.Image, adaptor: adaptors.Image) -> None:
    # FIXME: Ideally we could just use get_bounding_box(), which purports
    # to return a bounding box in parent space, however the local transform
    # of the Image is not used

    # No Transform
    image.transform = Transform()
    exp_bounds = np.asarray([[-0.5, -0.5, 0], [99.5, 199.5, 0]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Just Translation
    image.transform = Transform().translated((-10, -10))
    exp_bounds = np.asarray([[-10.5, -10.5, 0], [89.5, 189.5, 0]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Just Scaling
    image.transform = Transform().scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-0.25, -0.25, 0], [49.75, 99.75, 0]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Scaling then Translation
    image.transform = Transform().scaled((0.5, 0.5, 0.5)).translated((-10, -10))
    exp_bounds = np.asarray([[-10.25, -10.25, 0], [39.75, 89.75, 0]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)

    # Translation then Scaling
    image.transform = Transform().translated((-10, -10)).scaled((0.5, 0.5, 0.5))
    exp_bounds = np.asarray([[-5.25, -5.25, 0], [44.75, 94.75, 0]])
    bb = adaptor._pygfx_node.get_world_bounding_box()
    assert bb is not None
    assert np.array_equal(exp_bounds, bb)


def test_rgb(image: snx.Image, adaptor: adaptors.Image) -> None:
    """Tests RGB(A) images are correctly massaged to avoid shading errors."""
    # Assert a colormap can be used with 2D data
    image.data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    image.cmap = cmap.Colormap("red")
    np.testing.assert_array_equal(image.data, adaptor._pygfx_node.geometry.grid.data)  # pyright: ignore
    np.testing.assert_array_equal(
        cmap.Colormap("red").to_pygfx().texture.data,
        adaptor._pygfx_node.material.map.texture.data,  # pyright: ignore
    )

    # When the data changes to RGB, the adaptor's material map should be None
    image.data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    np.testing.assert_array_equal(image.data, adaptor._pygfx_node.geometry.grid.data)  # pyright: ignore
    assert adaptor._pygfx_node.material.map is None  # pyright: ignore

    # Even if the cmap is set, it should be ignored
    image.cmap = cmap.Colormap("blue")
    np.testing.assert_array_equal(image.data, adaptor._pygfx_node.geometry.grid.data)  # pyright: ignore
    assert adaptor._pygfx_node.material.map is None  # pyright: ignore

    # But it should snap to place if we go back to 2D data
    image.data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    np.testing.assert_array_equal(image.data, adaptor._pygfx_node.geometry.grid.data)  # pyright: ignore
    np.testing.assert_array_equal(
        cmap.Colormap("blue").to_pygfx().texture.data,
        adaptor._pygfx_node.material.map.texture.data,  # type: ignore
    )


def test_oversized_texture() -> None:
    """Demonstrates that textures exceeding GPU dimension limits are downsampled.

    When image data exceeds the GPU's max texture dimension, pygfx will raise a
    GLError at render time (err=1281, invalid value). The adaptor should detect
    oversized data and transparently downsample it (e.g. via a strided view) before
    uploading, rather than failing at render time.
    """
    import wgpu

    # Determine the GPU's max texture dimension
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    max_dim = device.limits["max-texture-dimension-2d"]

    # Create an image that exceeds the GPU limits
    oversized_shape = (1, max_dim + 1)
    image = snx.Image(
        data=np.zeros(oversized_shape, dtype=np.uint8), cmap=cmap.Colormap("viridis")
    )
    # Create an adaptor for it
    adaptor = image._get_adaptors(create=True)[0]
    assert isinstance(adaptor, adaptors.Image)
    # And check the texture was downsampled to fit within the GPU limits
    assert adaptor._texture.data.shape == (  # pyright: ignore
        oversized_shape[0],
        (oversized_shape[1] + 1) // 2,
    )
    # But ensure that the adaptor's transform is still correctly compensating for the
    # downsampling, so the image appears at the correct size
    np.testing.assert_almost_equal(
        adaptor._pygfx_node.get_world_bounding_box(), image.bounding_box
    )


def test_texture_reuse(image: snx.Image, adaptor: adaptors.Image) -> None:
    """Updating data with the same shape and dtype reuses the existing texture."""
    new_data = np.full(image.data.shape, 42, dtype=np.uint8)
    with patch(_TEXTURE_PATH, wraps=pygfx.Texture) as mock_texture:
        image.data = new_data
    assert mock_texture.call_count == 0
    np.testing.assert_array_equal(adaptor._texture.data, new_data)  # pyright: ignore


def test_texture_recreated_on_shape_change(
    image: snx.Image, adaptor: adaptors.Image
) -> None:
    """Updating data with a different shape creates a new texture."""
    existing_shape = adaptor._texture.data.shape  # pyright: ignore
    new_shape = (existing_shape[0] + 10, existing_shape[1] + 10)
    with patch(_TEXTURE_PATH, wraps=pygfx.Texture) as mock_texture:
        image.data = np.zeros(new_shape, dtype=np.uint8)
    assert mock_texture.call_count == 1


def test_texture_recreated_on_dtype_change(
    image: snx.Image, adaptor: adaptors.Image
) -> None:
    """Updating data with a different dtype creates a new texture."""
    assert adaptor._texture.data.dtype != np.float32  # pyright: ignore
    with patch(_TEXTURE_PATH, wraps=pygfx.Texture) as mock_texture:
        image.data = np.zeros((100, 100), dtype=np.float32)
    assert mock_texture.call_count == 1


def test_texture_recreated_on_ndim_change(
    image: snx.Image, adaptor: adaptors.Image
) -> None:
    """Switching between grayscale and RGB creates a new texture."""
    with patch(_TEXTURE_PATH, wraps=pygfx.Texture) as mock_texture:
        image.data = np.zeros((100, 100, 3), dtype=np.uint8)
    assert mock_texture.call_count == 1


def test_texture_buffer_not_source_array(
    image: snx.Image, adaptor: adaptors.Image
) -> None:
    """Reusing pygfx Textures opens the door to mutation - test it doesn't happen."""
    old_data_actual = image.data
    assert isinstance(old_data_actual, np.ndarray)
    old_data_expected = old_data_actual.copy()
    # Overwrite via same-shape update — triggers the in-place reuse path
    image.data = old_data_actual + 1
    # The first array must be unaffected
    assert np.array_equal(old_data_actual, old_data_expected)


@pytest.mark.parametrize(
    ("src_dtype", "expected_dtype"),
    [
        (np.float64, np.float32),
        (np.int64, np.float32),
        (np.uint64, np.float32),
    ],
)
def test_casting(src_dtype: np.dtype, expected_dtype: np.dtype) -> None:
    """Pygfx does not allow several dtypes. This test ensures those are casted."""
    rng = np.random.default_rng()
    if np.issubdtype(src_dtype, np.floating):
        data = rng.random((100, 100), dtype=src_dtype)
    else:
        data = rng.integers(0, 255, (100, 100), dtype=src_dtype)
    image = snx.Image(data=data, cmap=cmap.Colormap("viridis"))
    adaptor = get_adaptor_registry().get_adaptor(image, create=True)
    assert isinstance(adaptor, adaptors.Image)
    assert adaptor._texture.data.dtype == expected_dtype  # pyright: ignore
