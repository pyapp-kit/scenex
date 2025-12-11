from unittest.mock import MagicMock

import cmap
import numpy as np
import pytest

import scenex as snx
from scenex import Mesh
from scenex.app.events import Ray


@pytest.fixture
def mesh() -> snx.Mesh:
    vertices = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [0, 1, 0],  # 2
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
        ]
    )
    return snx.Mesh(
        vertices=vertices,
        faces=faces,
        color=snx.ColorModel(type="uniform", color=cmap.Color("red")),
    )


def test_bounding_box(mesh: snx.Mesh) -> None:
    exp_bounding_box = np.asarray(((0, 0, 0), (1, 1, 0)))
    assert np.array_equal(exp_bounding_box, mesh.bounding_box)


def test_passes_through(mesh: Mesh) -> None:
    # Check a ray that passes through the mesh hits
    ray = Ray(
        origin=(0.25, 0.25, 1), direction=(0, 0, -1), source=MagicMock(spec=snx.View)
    )
    assert mesh.passes_through(ray) == 1

    # Check a ray that grazes the left edge of the image hits
    ray = Ray(origin=(0, 0.5, 1), direction=(0, 0, -1), source=MagicMock(spec=snx.View))
    assert mesh.passes_through(ray) == 1

    # Check a ray that grazes the right edge of the image misses
    ray = Ray(
        origin=(0.5, 0.5, 1), direction=(0, 0, -1), source=MagicMock(spec=snx.View)
    )
    assert mesh.passes_through(ray) is None

    # Check a ray that does not pass through the image misses
    ray = Ray(
        origin=(-50, -50, 1), direction=(0, 0, -1), source=MagicMock(spec=snx.View)
    )
    assert mesh.passes_through(ray) is None

    # Check a ray that is perpendicular to the image misses
    ray = Ray(origin=(0, 0, 0), direction=(-1, 0, 0), source=MagicMock(spec=snx.View))
    assert mesh.passes_through(ray) is None
