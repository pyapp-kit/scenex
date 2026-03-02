from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx

from scenex.adaptors._base import LineAdaptor
from scenex.model._color import ColorModel, UniformColor, VertexColors

from ._node import Node

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from scenex import model


class Line(Node, LineAdaptor):
    """pygfx backend adaptor for a Line node."""

    _pygfx_node: pygfx.Line
    _material: pygfx.LineMaterial

    def __init__(self, line: model.Line, **backend_kwargs: Any) -> None:
        self._model = line
        self._material = pygfx.LineMaterial(
            thickness=line.width,
            aa=line.antialias,
            # This value has model render order win for coplanar objects
            depth_compare="<=",
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

    def _snx_set_color(self, arg: ColorModel) -> None:
        # Set the color first, and then the color mode
        # This order is probably smart because otherwise we'd be vulnerable to an
        # invalid intermediate state (e.g., setting color mode to 'vertex' when no
        # vertex colors)
        if isinstance(arg, UniformColor):
            self._material.color = arg.color.rgba
            self._material.color_mode = "uniform"
        elif isinstance(arg, VertexColors):
            self._geometry.colors = pygfx.resources.Buffer(
                np.asarray([a.rgba for a in arg.color], dtype=np.float32)
            )
            self._material.color_mode = "vertex"

    def _snx_set_width(self, arg: float) -> None:
        self._material.thickness = arg

    def _snx_set_antialias(self, arg: bool) -> None:
        self._material.aa = arg
