from __future__ import annotations

import cmap
import numpy as np
import pytest

import scenex as snx
import scenex.adaptors._pygfx as adaptors
from scenex.adaptors import get_adaptor_registry
from scenex.model._transform import Transform


@pytest.fixture
def image() -> snx.Image:
    return snx.Image(
        data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
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


@pytest.mark.parametrize(
    ("src_dtype", "expected_dtype"),
    [
        (np.float64, np.float32),
        (np.int64, np.int32),
        (np.uint64, np.uint32),
    ],
)
def test_downcasting(src_dtype: np.dtype, expected_dtype: np.dtype) -> None:
    """Pygfx does not allow 64-bit data. This test ensures 64-bit data is downcasted."""
    rng = np.random.default_rng()
    if np.issubdtype(src_dtype, np.floating):
        data = rng.random((100, 100), dtype=src_dtype)
    else:
        data = rng.integers(0, 255, (100, 100), dtype=src_dtype)
    image = snx.Image(data=data, cmap=cmap.Colormap("viridis"))
    adaptor = get_adaptor_registry().get_adaptor(image, create=True)
    assert isinstance(adaptor, adaptors.Image)
    assert adaptor._texture.data.dtype == expected_dtype  # pyright: ignore
