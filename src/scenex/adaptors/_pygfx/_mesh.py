from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import cmap
import numpy as np
import pygfx

from scenex.adaptors._base import MeshAdaptor

from ._node import Node

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from scenex import model

logger = logging.getLogger("scenex.adaptors.pygfx")


class Mesh(Node, MeshAdaptor):
    """pygfx backend adaptor for an Mesh node."""

    _pygfx_node: pygfx.Mesh

    def __init__(self, mesh: model.Mesh, **backend_kwargs: Any) -> None:
        self._model = mesh

        self._pygfx_node = pygfx.Mesh(
            material=pygfx.MeshBasicMaterial(),
            geometry=pygfx.Geometry(
                positions=pygfx.resources.Buffer(
                    np.asarray(mesh.vertices, dtype=np.float32)
                ),
                indices=pygfx.resources.Buffer(np.asarray(mesh.faces, dtype=np.uint32)),
            ),  # pyright: ignore
        )

        self._snx_set_color(mesh.color)

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
            self._pygfx_node.geometry.positions = pygfx.resources.Buffer(arg)  # pyright: ignore

    def _snx_set_faces(self, arg: ArrayLike) -> None:
        # Number of faces unchanged - reuse existing geometry for performance
        arg = np.asarray(arg, dtype=np.uint32)
        geom = self._pygfx_node.geometry
        indices: pygfx.resources.Buffer = geom.indices  # pyright: ignore
        if (data := indices.data) is not None and (arg.shape == data.shape):
            data[:, :] = arg
            indices.update_range()
        # Number of faces changed - must create a new Buffer
        else:
            self._pygfx_node.geometry.indices = pygfx.resources.Buffer(arg)  # pyright: ignore

    def _snx_set_color(self, arg: model.ColorModel) -> None:
        if arg.type == "uniform" and isinstance(arg.color, cmap.Color):
            self._pygfx_node.material.color_mode = "uniform"  # pyright: ignore
            self._pygfx_node.material.color = arg.color.rgba  # pyright: ignore
        elif arg.type == "vertex" and isinstance(arg.color, Sequence):
            self._pygfx_node.material.color_mode = "vertex"  # pyright: ignore
            self._pygfx_node.geometry.colors = pygfx.resources.Buffer(  # pyright: ignore
                np.asarray([c.rgba for c in arg.color], dtype=np.float32)
            )
        elif arg.type == "face":
            # TODO: per-face colors are tricky under this paradigm.
            # pygfx expects that if you provide N face colors, you have N triangles,
            # and vice-versa. This means that the intermediate state in between updating
            # the faces and the face colors will be invalid according to pygfx.
            # We likely will need some strategy for batching model updates.
            logger.warning("pygfx backend does not support per-face colors yet.")
