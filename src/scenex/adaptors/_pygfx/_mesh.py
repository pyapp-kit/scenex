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

    def __init__(self, mesh: model.Mesh, **backend_kwargs: Any) -> None:
        self._model = mesh
        self._pygfx_node = pygfx.Mesh(
            material=pygfx.MeshBasicMaterial(
                color=mesh.color.rgba if mesh.color else (1, 1, 1, 1)
            ),
            geometry=pygfx.Geometry(
                positions=np.asarray(mesh.vertices, dtype=np.float32),
                indices=np.asarray(mesh.faces, dtype=np.uint32),
            ),
        )

    def _snx_set_vertices(self, arg: ArrayLike) -> None:
        # Number of vertices unchanged - reuse existing geometry for performance
        arg = np.asarray(arg, dtype=np.float32)
        geom = self._pygfx_node.geometry
        positions: pygfx.resources.Buffer = geom.positions  # pyright: ignore
        if (data := positions.data) is not None and (arg.shape == data.shape):
            data[:, :] = arg
            positions.update_range()
        # Number of vertices changed - must create new geometry
        else:
            self._pygfx_node.geometry = pygfx.Geometry(
                positions=arg,
                indices=geom.indices,  # pyright: ignore
            )

    def _snx_set_faces(self, arg: ArrayLike) -> None:
        # Number of faces unchanged - reuse existing geometry for performance
        arg = np.asarray(arg, dtype=np.uint32)
        geom = self._pygfx_node.geometry
        indices: pygfx.resources.Buffer = geom.indices  # pyright: ignore
        if (data := indices.data) is not None and (arg.shape == data.shape):
            data[:, :] = arg
            indices.update_range()
        # Number of faces changed - must create new geometry
        else:
            self._pygfx_node.geometry = pygfx.Geometry(
                positions=geom.positions,  # pyright: ignore
                indices=arg,
            )

    def _snx_set_color(self, arg: cmap.Color) -> None:
        self._pygfx_node.material.color = arg.rgba  # pyright: ignore
