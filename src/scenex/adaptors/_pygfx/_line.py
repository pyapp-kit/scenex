from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import cmap
import numpy as np
import pygfx

from scenex.adaptors._base import LineAdaptor

from ._node import Node

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    import scenex as snx
    from scenex import model


class Line(Node, LineAdaptor):
    """pygfx backend adaptor for a Line node."""

    _pygfx_node: pygfx.Line
    _material: pygfx.LineMaterial

    def __init__(self, line: model.Line, **backend_kwargs: Any) -> None:
        self._model = line
        self._material = pygfx.LineMaterial(
            thickness=line.width,
        )
        self._geometry = pygfx.Geometry(
            positions=np.asarray(line.vertices, dtype=np.float32),
        )
        self._pygfx_node = pygfx.Line(
            geometry=self._geometry,
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

    def _snx_set_color(self, arg: snx.ColorModel) -> None:
        # TODO: There's got to be a more efficient way to do this...
        if arg.type == "uniform" and isinstance(arg.color, cmap.Color):
            self._geometry = self._pygfx_node.geometry = pygfx.Geometry(
                positions=self._geometry.positions.data,
            )
            self._material = self._pygfx_node.material = pygfx.LineMaterial(
                color_mode="uniform",
                thickness=self._material.thickness,
                color=arg.color.rgba,
            )
        elif arg.type == "vertex" and isinstance(arg.color, Sequence):
            self._geometry = self._pygfx_node.geometry = pygfx.Geometry(
                positions=self._geometry.positions.data,
                colors=np.asarray([a.rgba for a in arg.color], dtype=np.float32),
            )
            self._material = self._pygfx_node.material = pygfx.LineMaterial(
                color_mode="vertex",
                thickness=self._material.thickness,
            )

    def _snx_set_width(self, arg: float) -> None:
        self._material.thickness = arg
