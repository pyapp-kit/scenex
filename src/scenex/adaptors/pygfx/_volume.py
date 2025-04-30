from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap
    from numpy.typing import ArrayLike

    from scenex import model


class Volume(Node):
    """pygfx backend adaptor for a Volume node."""

    _pygfx_node: pygfx.Volume
    _material: pygfx.VolumeBasicMaterial
    _geometry: pygfx.Geometry

    def __init__(self, volume: model.Volume, **backend_kwargs: Any) -> None:
        self._snx_set_data(volume.data)
        self._snx_set_rendermode(volume.render_mode, volume.interpolation)
        self._pygfx_node = pygfx.Volume(self._geometry, self._material)

    def _snx_set_cmap(self, arg: Colormap) -> None:
        self._material.map = arg.to_pygfx()

    def _snx_set_clims(self, arg: tuple[float, float] | None) -> None:
        self._material.clim = arg

    def _snx_set_gamma(self, arg: float) -> None:
        self._material.gamma = arg

    def _snx_set_interpolation(self, arg: model.InterpolationMode) -> None:
        if arg == "bicubic":
            warnings.warn(
                "Bicubic interpolation not supported by pygfx - falling back to linear",
                RuntimeWarning,
                stacklevel=2,
            )
            arg = "linear"
        self._material.interpolation = arg

    def _create_texture(self, data: np.ndarray) -> pygfx.Texture:
        if data.ndim != 3:
            raise Exception("Volumes must be 3-dimensional")
        return pygfx.Texture(data, dim=data.ndim)

    def _snx_set_data(self, data: ArrayLike) -> None:
        self._texture = self._create_texture(np.asanyarray(data))
        self._geometry = pygfx.Geometry(grid=self._texture)

    def _snx_set_rendermode(
        self,
        data: model.RenderMode,
        interpolation: model.InterpolationMode | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {"depth_test": False}
        if interpolation is not None:
            kwargs["interpolation"] = interpolation
        elif self._material is not None:
            kwargs["interpolation"] = self._material.interpolation

        if data == "mip":
            self._material = pygfx.VolumeMipMaterial(**kwargs)
        elif data == "iso":
            self._material = pygfx.VolumeIsoMaterial(**kwargs)

        if hasattr(self, "_pygfx_node"):
            self._pygfx_node.material = self._material
