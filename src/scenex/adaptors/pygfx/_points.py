from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pygfx

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

    _pygfx_node: pygfx.Points

    def __init__(self, points: model.Points, **backend_kwargs: Any) -> None:
        # TODO: unclear whether get_view() is better here...
        coords = np.asarray(points.coords)
        n_coords = len(coords)

        # ensure (N, 3)
        if coords.shape[1] == 2:
            coords = np.column_stack((coords, np.zeros(coords.shape[0])))

        geo_kwargs = {}
        if points.face_color is not None:
            colors = np.tile(np.asarray(points.face_color), (n_coords, 1))
            geo_kwargs["colors"] = colors.astype(np.float32)

        # TODO: not sure whether/how pygfx implements all the other properties

        self._geometry = pygfx.Geometry(
            positions=coords.astype(np.float32),
            sizes=np.full(n_coords, points.size, dtype=np.float32),
            **geo_kwargs,
        )
        self._material = pygfx.PointsMaterial(
            size=points.size,  # pyright: ignore[reportArgumentType]
            size_space=SPACE_MAP[points.scaling],
            aa=points.antialias > 0,
            opacity=points.opacity,
            color_mode="vertex",
            size_mode="vertex",
        )
        self._pygfx_node = pygfx.Points(self._geometry, self._material)

    def _snx_set_coords(self, coords: npt.NDArray) -> None: ...

    def _snx_set_size(self, size: float) -> None: ...

    def _snx_set_face_color(self, face_color: Color) -> None: ...

    def _snx_set_edge_color(self, edge_color: Color) -> None: ...

    def _snx_set_edge_width(self, edge_width: float) -> None: ...

    def _snx_set_symbol(self, symbol: str) -> None: ...

    def _snx_set_scaling(self, scaling: str) -> None: ...

    def _snx_set_antialias(self, antialias: float) -> None: ...

    def _snx_set_opacity(self, arg: float) -> None: ...
