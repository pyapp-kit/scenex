from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx

from scenex.adaptors._base import LineAdaptor

from ._node import Node

if TYPE_CHECKING:
    import cmap
    from numpy.typing import ArrayLike

    from scenex import model

logger = logging.getLogger("scenex.adaptors.pygfx")


class Line(Node, LineAdaptor):
    """pygfx backend adaptor for a Line node."""

    _pygfx_node: pygfx.Line
    _material: pygfx.LineMaterial

    def __init__(self, line: model.Line, **backend_kwargs: Any) -> None:
        self._model = line
        self._material = pygfx.LineMaterial(
            color=line.color.rgba if line.color else (1, 1, 1, 1),
            thickness=line.width,
        )
        self._pygfx_node = pygfx.Line(
            geometry=pygfx.Geometry(
                positions=np.asarray(line.vertices, dtype=np.float32),
            ),
            material=self._material,
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
            self._pygfx_node.geometry = pygfx.Geometry(positions=arg)

    def _snx_set_color(self, arg: cmap.Color) -> None:
        self._material.color = arg.rgba

    def _snx_set_width(self, arg: float) -> None:
        self._material.thickness = arg
