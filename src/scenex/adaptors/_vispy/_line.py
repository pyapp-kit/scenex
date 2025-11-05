from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import cmap
import numpy as np
import vispy.scene
import vispy.visuals

from scenex.adaptors._base import LineAdaptor

from ._node import Node

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    import scenex as snx
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
            antialias=line.antialias > 0,
        )
        self._snx_set_color(line.color)

    def _snx_set_vertices(self, arg: ArrayLike) -> None:
        self._vispy_node.set_data(pos=np.asarray(arg, dtype=np.float32))

    def _snx_set_color(self, arg: snx.ColorModel) -> None:
        if arg.type == "uniform" and isinstance(arg.color, cmap.Color):
            self._vispy_node.set_data(color=arg.color.hex)
        elif isinstance(arg.color, Sequence):
            self._vispy_node.set_data(color=[a.hex for a in arg.color])

    def _snx_set_width(self, arg: float) -> None:
        self._vispy_node.set_data(width=arg)
