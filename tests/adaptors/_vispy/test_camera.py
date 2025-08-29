from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from vispy.scene.cameras import BaseCamera

import scenex as snx
import scenex.adaptors._vispy as adaptors
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
    # Centered at [0, 0], top left [-1, -1], bottom right [1, 1]
    assert model.transform == Transform()
    assert model.projection == projections.orthographic(2, 2, 2)

    node = adaptor._vispy_node
    assert isinstance(node, BaseCamera)

    w, h = node.viewbox.size
    # Vispy wants to map [-1, 1] to [0, 0]
    # Vispy wants to map [1, -1] to [w, h]
    exp_tform_mat = np.asarray(
        [
            [w / 2, 0, 0, 0],
            [0, -h / 2, 0, 0],
            [0, 0, -1, 0],
            [w / 2, h / 2, 0, 1],
        ]
    )
    assert np.array_equal(node.transform.matrix, exp_tform_mat)  # pyright: ignore[reportAttributeAccessIssue]


def test_transform_translate(camera: tuple[snx.Camera, adaptors.Camera]) -> None:
    model, adaptor = camera

    node = adaptor._vispy_node
    assert isinstance(node, BaseCamera)

    # Move the camera
    model.transform = Transform().translated((1, 1))

    w, h = node.viewbox.size
    # Vispy wants to map [0, 2] to [0, 0]
    # Vispy wants to map [2, 0] to [w, h]
    exp_tform_mat = np.asarray(
        [
            [w / 2, 0, 0, 0],
            [0, -h / 2, 0, 0],
            [0, 0, -1, 0],
            [0, h, 0, 1],
        ]
    )
    assert np.array_equal(node.transform.matrix, exp_tform_mat)  # pyright: ignore[reportAttributeAccessIssue]


def test_transform_scale(camera: tuple[snx.Camera, adaptors.Camera]) -> None:
    model, adaptor = camera

    node = adaptor._vispy_node
    assert isinstance(node, BaseCamera)

    # Widen the projection matrix
    model.projection = projections.orthographic(4, 4, 4)

    w, h = node.viewbox.size
    # Vispy wants to map [-2, 2] to [0, 0]
    # Vispy wants to map [2, -2] to [10, 10]
    exp_tform_mat = np.asarray(
        [
            [w / 4, 0, 0, 0],
            [0, -h / 4, 0, 0],
            [0, 0, -1 / 2, 0],
            [w / 2, h / 2, 0, 1],
        ]
    )
    assert np.array_equal(node.transform.matrix, exp_tform_mat)  # pyright: ignore[reportAttributeAccessIssue]
