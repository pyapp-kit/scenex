from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx.cameras as pygfx
import pytest

import scenex as snx
import scenex.adaptors._pygfx as adaptors
from scenex.adaptors import get_adaptor_registry
from scenex.model._transform import Transform
from scenex.utils import projections

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def camera() -> Generator[tuple[snx.Camera, adaptors.Camera], None, None]:
    view = snx.View(camera=snx.Camera())
    adaptor = get_adaptor_registry().get_adaptor(view.camera, create=True)
    get_adaptor_registry().get_adaptor(view, create=True)
    assert isinstance(adaptor, adaptors.Camera)
    # TODO: Do we need to hold on to a view ref?
    yield (view.camera, adaptor)


def test_transform_defaults(camera: tuple[snx.Camera, adaptors.Camera]) -> None:
    model, adaptor = camera

    node = adaptor._pygfx_node
    assert isinstance(node, pygfx.Camera)
    # Centered at [0, 0], top left [-1, -1], bottom right [1, 1]
    assert model.transform == Transform()
    assert model.projection == projections.orthographic(2, 2, 2)

    # Assert internal pygfx matrices match the model matrices
    # Note that pygfx matrices are transposes of scenex matrices
    assert np.array_equal(node.local.matrix.T, Transform())
    assert np.array_equal(node.projection_matrix.T, projections.orthographic(2, 2, 2))  # pyright: ignore[reportAttributeAccessIssue]


def test_transform_translate(camera: tuple[snx.Camera, adaptors.Camera]) -> None:
    model, adaptor = camera

    node = adaptor._pygfx_node
    assert isinstance(node, pygfx.Camera)

    # Move the camera
    model.transform = Transform().translated((0.5, 0.5))

    # Assert internal pygfx matrices match the expected model matrices
    # Note that pygfx matrices are transposes of scenex matrices
    assert np.array_equal(node.local.matrix.T, Transform().translated((0.5, 0.5)))
    assert np.array_equal(node.projection_matrix.T, projections.orthographic(2, 2, 2))  # pyright: ignore[reportAttributeAccessIssue]


def test_transform_scale(camera: tuple[snx.Camera, adaptors.Camera]) -> None:
    model, adaptor = camera

    node = adaptor._pygfx_node
    assert isinstance(node, pygfx.Camera)

    # Widen the projection matrix
    model.projection = projections.orthographic(4, 4, 4)

    # Assert internal pygfx matrices match the expected model matrices
    # Note that pygfx matrices are transposes of scenex matrices
    assert np.array_equal(node.local.matrix.T, Transform())
    assert np.array_equal(node.projection_matrix.T, projections.orthographic(4, 4, 4))  # pyright: ignore[reportAttributeAccessIssue]
