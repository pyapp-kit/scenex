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
def camera() -> tuple[snx.Camera, adaptors.Camera]:
    model_cam = snx.Camera()
    adaptor = get_adaptor_registry().get_adaptor(model_cam, create=True)
    assert isinstance(adaptor, adaptors.Camera)
    return (model_cam, adaptor)


@pytest.fixture
def view_camera() -> Generator[tuple[snx.Camera, adaptors.Camera], None, None]:
    view = snx.View(camera=snx.Camera())
    adaptor = get_adaptor_registry().get_adaptor(view.camera, create=True)
    get_adaptor_registry().get_adaptor(view, create=True)
    assert isinstance(adaptor, adaptors.Camera)
    # TODO: Do we need to hold on to a view ref?
    yield (view.camera, adaptor)


@pytest.fixture
def adaptor(camera: snx.Camera) -> adaptors.Camera:
    adaptor = get_adaptor_registry().get_adaptor(camera, create=True)
    assert isinstance(adaptor, adaptors.Camera)
    return adaptor


def test_transform_with_view(view_camera: tuple[snx.Camera, adaptors.Camera]) -> None:
    camera, adaptor = view_camera

    node = adaptor._vispy_node
    assert isinstance(node, BaseCamera)
    # Centered at [0, 0], top left [-1, -1], bottom right [1, 1]
    identity_tform = Transform()
    assert camera.transform == identity_tform
    # Vispy wants to map [-1, -1] to [0, 0]
    # Vispy wants to map [1, 1] to [10, 10]
    exp_tform_mat = np.asarray(
        [
            [5, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 1, 0],
            [5, 5, 0, 1],
        ]
    )
    assert np.array_equal(node.transform.matrix, exp_tform_mat)  # pyright: ignore[reportAttributeAccessIssue]

    # Move the camera
    camera.transform = Transform().translated((1, 1))
    # Vispy wants to map [0, 0] to [0, 0]
    # Vispy wants to map [2, 2] to [10, 10]
    exp_tform_mat = np.asarray(
        [
            [5, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    assert np.array_equal(node.transform.matrix, exp_tform_mat)  # pyright: ignore[reportAttributeAccessIssue]


def test_projection_with_view(view_camera: tuple[snx.Camera, adaptors.Camera]) -> None:
    camera, adaptor = view_camera

    node = adaptor._vispy_node
    assert isinstance(node, BaseCamera)
    # Centered at [0, 0], top left [-1, -1], bottom right [1, 1]
    identity_tform = Transform()
    assert camera.projection == identity_tform
    # Vispy wants to map [-1, -1] to [0, 0]
    # Vispy wants to map [1, 1] to [10, 10]
    exp_tform_mat = np.asarray(
        [
            [5, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 1, 0],
            [5, 5, 0, 1],
        ]
    )
    assert np.array_equal(node.transform.matrix, exp_tform_mat)  # pyright: ignore[reportAttributeAccessIssue]

    # Widen the projection matrix
    camera.projection = projections.orthographic(4, 4)
    # Vispy wants to map [-2, -2] to [0, 0]
    # Vispy wants to map [2, 2] to [10, 10]
    exp_tform_mat = np.asarray(
        [
            [2.5, 0, 0, 0],
            [0, 2.5, 0, 0],
            [0, 0, -1, 0],
            [5, 5, 0, 1],
        ]
    )
    assert np.array_equal(node.transform.matrix, exp_tform_mat)  # pyright: ignore[reportAttributeAccessIssue]
