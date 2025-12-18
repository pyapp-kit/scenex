from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pygfx

from scenex.adaptors._base import PointsAdaptor
from scenex.model._color import ColorModel, UniformColor, VertexColors

from ._node import Node

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

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
    _material: pygfx.PointsMarkerMaterial
    _geometry: pygfx.Geometry

    def __init__(self, points: model.Points, **backend_kwargs: Any) -> None:
        self._model = points

        self._material = pygfx.PointsMarkerMaterial(
            size=points.size,  # pyright: ignore[reportArgumentType]
            size_space=SPACE_MAP[points.scaling],
            aa=points.antialias > 0,
            edge_width=points.edge_width,
            opacity=points.opacity,
        )
        if isinstance(points.face_color, UniformColor):
            self._material.color = points.face_color.color.rgba
        if isinstance(points.edge_color, UniformColor):
            self._material.edge_color = points.edge_color.color.rgba

        # Fill this in empty for now; will be populated in _snx_set_coords
        self._geometry = pygfx.Geometry(positions=np.zeros((1, 3), dtype=np.float32))
        self._snx_set_coords(points.coords)

        self._pygfx_node = pygfx.Points(self._geometry, self._material)

    def _snx_set_coords(self, coords: npt.NDArray | None) -> None:
        # ensure (N, 3)
        if coords is None or coords.size == 0:
            coords = np.zeros((0, 3), dtype=np.float32)
        elif coords.shape[1] == 2:
            coords = np.column_stack((coords, np.zeros(coords.shape[0])))
        # Coerce dtypes to float32 - suprisingly pygfx is sensitive to this
        coords = coords.astype(np.float32)
        # Update existing buffer if possible for performance
        positions = self._geometry.positions
        if (data := positions.data) is not None and (coords.shape == data.shape):
            data[:, :] = coords
            positions.update_range()
        # Otherwise create a new buffer
        else:
            self._geometry.positions = pygfx.resources.Buffer(coords)

    def _snx_set_size(self, size: float) -> None:
        self._material.size = size  # pyright: ignore

    def _snx_set_face_color(self, arg: ColorModel) -> None:
        if isinstance(arg, UniformColor):
            self._material.color_mode = "uniform"
            self._material.color = arg.color.rgba
        elif isinstance(arg, VertexColors):
            self._material.color_mode = "vertex"
            self._geometry.colors = pygfx.resources.Buffer(
                np.asarray([c.rgba for c in arg.color], dtype=np.float32)
            )

    def _snx_set_edge_color(self, arg: ColorModel) -> None:
        if isinstance(arg, UniformColor):
            self._material.edge_color_mode = "uniform"
            self._material.edge_color = arg.color.rgba
        elif isinstance(arg, VertexColors):
            self._material.edge_color_mode = "vertex"
            self._geometry.edge_colors = pygfx.resources.Buffer(
                np.asarray([c.rgba for c in arg.color], dtype=np.float32)
            )

    def _snx_set_edge_width(self, edge_width: float) -> None:
        self._material.edge_width = edge_width

    def _snx_set_symbol(self, symbol: str) -> None: ...

    def _snx_set_scaling(self, scaling: model.ScalingMode) -> None:
        self._material.size_space = SPACE_MAP[scaling]

    def _snx_set_antialias(self, antialias: float) -> None:
        self._material.aa = antialias > 0

    def _snx_set_opacity(self, arg: float) -> None:
        self._material.opacity = arg
