from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import vispy.color
import vispy.scene
import vispy.visuals

from ._node import Node

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    from cmap import Color

    from scenex import model

SPACE_MAP: Mapping[model.ScalingMode, Literal["model", "screen", "world"]] = {
    True: "world",
    False: "screen",
    "fixed": "screen",
    "scene": "world",
    "visual": "model",
}


class Points(Node):
    """Vispy backend adaptor for an Points node."""

    _model: model.Points
    _vispy_node: vispy.visuals.MarkersVisual

    def __init__(self, points: model.Points, **backend_kwargs: Any) -> None:
        self._model = points
        self._vispy_node = vispy.scene.Markers(
            pos=np.asarray(points.coords),
            symbol=points.symbol,
            scaling=points.scaling,  # pyright: ignore
            antialias=points.antialias,  # pyright: ignore
            edge_color=points.edge_color,
            edge_width=points.edge_width,
            face_color=points.face_color,
        )

    def _snx_set_coords(self, coords: npt.NDArray) -> None:
        # TODO: Will this overwrite our other parameters?
        self._update_vispy_data()

    def _snx_set_size(self, size: float) -> None:
        self._update_vispy_data()

    def _snx_set_face_color(self, face_color: Color) -> None:
        self._update_vispy_data()

    def _snx_set_edge_color(self, edge_color: Color) -> None:
        self._update_vispy_data()

    def _snx_set_edge_width(self, edge_width: float) -> None:
        self._update_vispy_data()

    def _snx_set_symbol(self, symbol: str) -> None:
        self._update_vispy_data()

    def _snx_set_scaling(self, scaling: str) -> None:
        self._update_vispy_data()

    def _snx_set_antialias(self, antialias: float) -> None:
        self._vispy_node.antialias = antialias

    def _snx_set_opacity(self, arg: float) -> None:
        self._vispy_node.alpha = arg

    def _update_vispy_data(self) -> None:
        # All of the _snx setters that deal with the "set_data" method pass through
        # here. We must remember and pass through all of these parameters every time,
        # or the node will revert to the defaults.
        if self._model.edge_color and self._model.edge_color.name:
            edge_color = self._model.edge_color.name
        else:
            edge_color = "black"

        if self._model.face_color and self._model.face_color.name:
            face_color = self._model.face_color.name
        else:
            face_color = "white"

        self._vispy_node.set_data(
            pos=np.asarray(self._model.coords),
            symbol=self._model.symbol,
            scaling=self._model.scaling,  # pyright: ignore
            face_color=face_color,
            edge_color=edge_color,
            edge_width=self._model.edge_width,
        )
