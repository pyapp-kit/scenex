from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx

from scenex.adaptors._base import MeshAdaptor

from ._node import Node

if TYPE_CHECKING:
    import cmap
    from numpy.typing import ArrayLike

    from scenex import model

logger = logging.getLogger("scenex.adaptors.pygfx")


class Mesh(Node, MeshAdaptor):
    """pygfx backend adaptor for an Mesh node."""

    _pygfx_node: pygfx.Mesh
    _material: pygfx.MeshBasicMaterial
    _geometry: pygfx.Geometry

    def __init__(self, mesh: model.Mesh, **backend_kwargs: Any) -> None:
        self._model = mesh
        self._material = pygfx.MeshBasicMaterial(
            color=mesh.color.rgba if mesh.color else (1, 1, 1, 1)
        )
        self._geometry = pygfx.Geometry(
            positions=np.asarray(mesh.vertices, dtype=np.float32),
            indices=np.asarray(mesh.faces, dtype=np.int32),
        )
        self._pygfx_node = pygfx.Mesh(self._geometry, self._material)

    def _snx_set_vertices(self, arg: ArrayLike) -> None:
        return
        self._geometry.positions = np.asarray(arg)

    def _snx_set_faces(self, arg: ArrayLike) -> None:
        return
        self._geometry.indices = np.asarray(arg)

    def _snx_set_color(self, arg: cmap.Color) -> None:
        return
        self._material.color = arg.rgba
