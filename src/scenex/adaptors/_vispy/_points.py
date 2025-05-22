from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
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

    # FIXME: Better understand the issue
    _vispy_node: vispy.visuals.MarkersVisual  # pyright: ignore

    def __init__(self, points: model.Points, **backend_kwargs: Any) -> None:
        # TODO: unclear whether get_view() is better here...
        coords = np.asarray(points.coords)
        # ensure (N, 3)
        if coords.shape[1] == 2:
            coords = np.column_stack((coords, np.zeros(coords.shape[0])))

        self._vispy_node = vispy.scene.Markers(
            pos=np.asarray(points.coords),
            symbol=points.symbol,
            scaling=points.scaling,  # pyright: ignore
            antialias=points.antialias,  # pyright: ignore
            edge_color=points.edge_color,
            edge_width=points.edge_width,
            face_color=points.face_color,
        )

    def _snx_set_coords(self, coords: npt.NDArray) -> None: ...

    def _snx_set_size(self, size: float) -> None: ...

    def _snx_set_face_color(self, face_color: Color) -> None: ...

    def _snx_set_edge_color(self, edge_color: Color) -> None: ...

    def _snx_set_edge_width(self, edge_width: float) -> None: ...

    def _snx_set_symbol(self, symbol: str) -> None: ...

    def _snx_set_scaling(self, scaling: str) -> None: ...

    def _snx_set_antialias(self, antialias: float) -> None: ...

    def _snx_set_opacity(self, arg: float) -> None: ...
