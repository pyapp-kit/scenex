from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx

from scenex.adaptors._base import VolumeAdaptor
from scenex.adaptors._pygfx._image import DOWNCASTS

from ._node import Node

logger = logging.getLogger("scenex.adaptors.pygfx")

if TYPE_CHECKING:
    from cmap import Colormap
    from numpy.typing import ArrayLike

    from scenex import model


class Volume(Node, VolumeAdaptor):
    """pygfx backend adaptor for a Volume node."""

    _pygfx_node: pygfx.Volume
    _material: pygfx.VolumeBasicMaterial
    _geometry: pygfx.Geometry

    def __init__(self, volume: model.Volume, **backend_kwargs: Any) -> None:
        self._snx_set_data(volume.data)
        self._snx_set_render_mode(volume.render_mode, volume.interpolation)
        self._pygfx_node = pygfx.Volume(self._geometry, self._material)

    def _snx_set_cmap(self, arg: Colormap) -> None:
        self._material.map = arg.to_pygfx()

    def _snx_set_clims(self, arg: tuple[float, float] | None) -> None:
        self._material.clim = arg

    def _snx_set_gamma(self, arg: float) -> None:
        self._material.gamma = arg

    def _snx_set_interpolation(self, arg: model.InterpolationMode) -> None:
        if arg == "bicubic":
            logger.warning(
                "Bicubic interpolation not supported by pygfx - falling back to linear",
            )
            arg = "linear"
        self._material.interpolation = arg

    def _create_texture(self, data: np.ndarray) -> pygfx.Texture:
        if data.ndim != 3:
            raise Exception("Volumes must be 3-dimensional")
        if data.dtype in DOWNCASTS:
            cast_to = DOWNCASTS[data.dtype]
            # pygfx doesn't support 64-bit dtypes; downcast transparently.
            # The user hasn't done anything wrong — this is a backend limitation.
            logger.warning(
                "Downcasting volume data from %s to %s for pygfx compatibility",
                data.dtype.name,
                cast_to.name,
            )
            data = data.astype(cast_to)
        return pygfx.Texture(data, dim=data.ndim)

    def _snx_set_data(self, data: ArrayLike) -> None:
        self._texture = self._create_texture(np.asanyarray(data))
        self._geometry = pygfx.Geometry(grid=self._texture)
        if hasattr(self, "_pygfx_node"):
            self._pygfx_node.geometry = self._geometry

    def _snx_set_render_mode(
        self,
        data: model.RenderMode,
        interpolation: model.InterpolationMode | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {"depth_test": False}
        if interpolation is not None:
            kwargs["interpolation"] = interpolation
        elif self._material is not None:
            kwargs["interpolation"] = self._material.interpolation
            kwargs["clim"] = self._material.clim
            kwargs["map"] = self._material.map
            kwargs["gamma"] = self._material.gamma
            kwargs["opacity"] = self._material.opacity
            # alpha_mode parameter introduced in pygfx 0.13.0
            if hasattr(self._material, "alpha_mode"):
                kwargs["alpha_mode"] = self._material.alpha_mode

        if data == "mip":
            self._material = pygfx.VolumeMipMaterial(**kwargs)
        elif data == "iso":
            self._material = pygfx.VolumeIsoMaterial(**kwargs)

        if hasattr(self, "_pygfx_node"):
            self._pygfx_node.material = self._material
