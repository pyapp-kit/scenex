from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import vispy.scene
import vispy.visuals

from scenex.adaptors._base import PointsAdaptor
from scenex.model._color import UniformColor, VertexColors

from ._node import Node

if TYPE_CHECKING:
    import numpy.typing as npt

    from scenex import model


class Points(Node, PointsAdaptor):
    """Vispy backend adaptor for an Points node."""

    _model: model.Points
    _vispy_node: vispy.visuals.MarkersVisual

    def __init__(self, points: model.Points, **backend_kwargs: Any) -> None:
        self._model = points
        self._vispy_node = vispy.scene.Markers()
        self._update_vispy_data()

    def _snx_set_vertices(self, vertices: npt.NDArray) -> None:
        self._update_vispy_data()

    def _snx_set_size(self, size: float) -> None:
        # FIXME: There seems to be a limit on the maximum size of points in vispy.
        # i.e. suppose you have "scene" scaling, and an orthographic camera
        # showing [-2, 2] in x and y.  If you set the size to 0.5 (taking up 1/4
        # of the view), things work nicely. But if you set the size to 2 (taking up the
        # entire width/height) or 1 for that matter, the point does not actually get
        # that big on screen.
        self._update_vispy_data()

    def _snx_set_face_color(self, face_color: model.ColorModel) -> None:
        self._update_vispy_data()

    def _snx_set_edge_color(self, edge_color: model.ColorModel) -> None:
        self._update_vispy_data()

    def _snx_set_edge_width(self, edge_width: float) -> None:
        self._update_vispy_data()

    def _snx_set_symbol(self, symbol: str) -> None:
        self._update_vispy_data()

    def _snx_set_scaling(self, scaling: model.ScalingMode) -> None:
        self._vispy_node.scaling = scaling
        self._update_vispy_data()

    def _snx_set_antialias(self, antialias: bool) -> None:
        self._vispy_node.antialias = 1.0 if antialias else 0.0

    def _snx_set_opacity(self, arg: float) -> None:
        self._vispy_node.alpha = arg

    def _update_vispy_data(self) -> None:
        # All of the _snx setters that deal with the "set_data" method pass through
        # here. We must remember and pass through all of these parameters every time,
        # or the node will revert to the defaults.
        edge_color: str | list[str]
        if isinstance(self._model.edge_color, UniformColor):
            edge_color = self._model.edge_color.color.hex
        elif isinstance(self._model.edge_color, VertexColors):
            edge_color = [c.hex for c in self._model.edge_color.color]
        else:
            raise NotImplementedError("Unsupported edge color model")

        face_color: str | list[str]
        if isinstance(self._model.face_color, UniformColor):
            face_color = self._model.face_color.color.hex
        elif isinstance(self._model.face_color, VertexColors):
            face_color = [c.hex for c in self._model.face_color.color]
        else:
            raise NotImplementedError("Unsupported face color model")

        self._vispy_node.set_data(
            pos=np.asarray(self._model.vertices),
            size=self._model.size,
            symbol=self._model.symbol,
            face_color=face_color,  # pyright: ignore
            edge_color=edge_color,  # pyright: ignore
            edge_width=self._model.edge_width,
        )
        self._vispy_node.update()
