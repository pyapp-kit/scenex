from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import vispy.scene
import vispy.visuals

import scenex as snx
from scenex.adaptors._base import LineAdaptor

from ._node import Node

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from scenex import model

logger = logging.getLogger("scenex.adaptors.vispy")


class Line(Node, LineAdaptor):
    """vispy backend adaptor for a Line node."""

    _vispy_node: vispy.visuals.LineVisual

    def __init__(self, line: model.Line, **backend_kwargs: Any) -> None:
        self._model = line
        self._vispy_node = vispy.scene.Line(
            pos=np.asarray(line.vertices, dtype=np.float32),
            width=int(line.width),
            antialias=line.antialias,
        )
        self._snx_set_color(line.color)

    def _snx_set_vertices(self, arg: ArrayLike) -> None:
        self._vispy_node.set_data(pos=np.asarray(arg, dtype=np.float32))

    def _snx_set_color(self, arg: snx.ColorModel) -> None:
        if isinstance(arg, snx.UniformColor):
            self._vispy_node.set_data(color=arg.color.hex)
        elif isinstance(arg, snx.VertexColors):
            self._vispy_node.set_data(color=[a.hex for a in arg.color])

    def _snx_set_width(self, arg: float) -> None:
        self._vispy_node.set_data(width=arg)

    def _snx_set_antialias(self, arg: bool) -> None:
        self._vispy_node.antialias = arg
