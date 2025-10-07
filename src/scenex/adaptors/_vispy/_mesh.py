from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import vispy.color
import vispy.scene
import vispy.visuals

from scenex.adaptors._base import MeshAdaptor

from ._node import Node

if TYPE_CHECKING:
    import cmap
    from numpy.typing import ArrayLike

    from scenex import model

logger = logging.getLogger("scenex.adaptors.pygfx")


class Mesh(Node, MeshAdaptor):
    """vispy backend adaptor for an Mesh node."""

    _vispy_node: vispy.visuals.MeshVisual

    def __init__(self, mesh: model.Mesh, **backend_kwargs: Any) -> None:
        self._model = mesh
        self._vispy_node = vispy.scene.Mesh(
            color=vispy.color.Color(mesh.color.hex if mesh.color else "#ffffff"),
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32),
        )

    def _snx_set_vertices(self, arg: ArrayLike) -> None:
        self._vispy_node.set_data(vertices=np.asarray(arg), faces=self._model.faces)

    def _snx_set_faces(self, arg: ArrayLike) -> None:
        self._vispy_node.set_data(
            vertices=self._model.vertices, faces=np.asarray(arg, dtype=np.uint32)
        )

    def _snx_set_color(self, arg: cmap.Color) -> None:
        self._vispy_node.color = arg.hex
