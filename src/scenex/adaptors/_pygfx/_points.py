from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import cmap
import numpy as np
import pygfx

from scenex.adaptors._base import PointsAdaptor

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


class Points(Node, PointsAdaptor):
    """Vispy backend adaptor for an Points node."""

    _pygfx_node: pygfx.Points
    _material: pygfx.PointsMaterial
    _geometry: pygfx.Geometry

    def __init__(self, points: model.Points, **backend_kwargs: Any) -> None:
        self._model = points

        self._material = pygfx.PointsMaterial(
            size=points.size,  # pyright: ignore[reportArgumentType]
            size_space=SPACE_MAP[points.scaling],
            aa=points.antialias > 0,
            opacity=points.opacity,
            color_mode="vertex",
            size_mode="vertex",
        )
        self._pygfx_node = pygfx.Points(None, self._material)
        self._snx_set_coords(points.coords)

    def _create_geometry(self, coords: npt.NDArray | None) -> pygfx.Geometry:
        # TODO: unclear whether get_view() is better here...
        coords = np.asarray(coords)
        n_coords = len(coords)

        # ensure (N, 3)
        if coords.shape[1] == 2:
            coords = np.column_stack((coords, np.zeros(coords.shape[0])))

        geo_kwargs = {}
        if self._model.face_color is not None:
            colors = np.tile(np.asarray(self._model.face_color), (n_coords, 1))
            geo_kwargs["colors"] = colors.astype(np.float32)

        return pygfx.Geometry(
            positions=coords.astype(np.float32),
            sizes=np.full(n_coords, self._model.size, dtype=np.float32),
            **geo_kwargs,
        )

    def _snx_set_coords(self, coords: npt.NDArray | None) -> None:
        self._pygfx_node.geometry = self._create_geometry(coords)

    def _snx_set_size(self, size: float) -> None:
        n_coords = len(self._model.coords)
        sizes = np.full(n_coords, self._model.size, dtype=np.float32)
        self._pygfx_node.geometry.sizes = pygfx.Buffer(sizes)  # pyright: ignore[reportOptionalMemberAccess]

    def _color_buffer(self, color: Color | None) -> pygfx.Buffer:
        if color is None:
            color = cmap.Color("transparent")
        n_coords = len(self._model.coords)
        colors = np.tile(np.asarray(color), (n_coords, 1))
        return pygfx.Buffer(colors.astype(np.float32))

    def _snx_set_face_color(self, face_color: Color | None) -> None:
        self._pygfx_node.geometry.colors = self._color_buffer(face_color)  # pyright: ignore[reportOptionalMemberAccess]

    def _snx_set_edge_color(self, edge_color: Color | None) -> None:
        self._pygfx_node.geometry.edge_color = self._color_buffer(edge_color)  # pyright: ignore[reportOptionalMemberAccess]

    def _snx_set_edge_width(self, edge_width: float) -> None: ...

    def _snx_set_symbol(self, symbol: str) -> None: ...

    def _snx_set_scaling(self, scaling: model.ScalingMode) -> None:
        self._material.size_space = SPACE_MAP[scaling]

    def _snx_set_antialias(self, antialias: float) -> None: ...

    def _snx_set_opacity(self, arg: float) -> None:
        self._material.opacity = arg
