from __future__ import annotations

import numpy as np
import pygfx
import pytest

import scenex as snx
import scenex.adaptors._pygfx as adaptors
from scenex.adaptors._auto import get_adaptor_registry
from scenex.model import BlendMode, Transform


@pytest.fixture
def volume() -> snx.Volume:
    return snx.Volume(
        data=np.random.randint(0, 255, (100, 100, 100), dtype=np.uint8),
    )


@pytest.fixture
def adaptor(volume: snx.Volume) -> adaptors.Volume:
    adaptor = get_adaptor_registry().get_adaptor(volume, create=True)
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
