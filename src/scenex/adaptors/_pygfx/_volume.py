from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx

from scenex.adaptors._base import VolumeAdaptor
from scenex.adaptors._pygfx._image import (
    _coerce_data,
)

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

    def __init__(self, volume: model.Volume, **backend_kwargs: Any) -> None:
        self._model = volume
        self._snx_set_render_mode(volume.render_mode, volume.interpolation)
        self._pygfx_node = pygfx.Volume(material=self._material)
        self._snx_set_data(volume.data)

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

    def _snx_set_transform(self, arg: model.Transform) -> None:
        if not hasattr(self, "_pygfx_node"):
            return  # _snx_set_data hasn't run yet to set initial factors
        # Compensate for downscaled textures
        # NOTE that image axes are ZYX in model space
        # but the pygfx Texture shape will be (X, Y, Z).
        x_fac = self._model.data.shape[2] / self._texture.size[0]
        y_fac = self._model.data.shape[1] / self._texture.size[1]
        z_fac = self._model.data.shape[0] / self._texture.size[2]
        self._pygfx_node.local.matrix = (
            arg.scaled((x_fac, y_fac, z_fac))
            .translated((0.5 * (x_fac - 1), 0.5 * (y_fac - 1), 0.5 * (z_fac - 1)))
            .root.T
        )

    def _snx_set_data(self, data: ArrayLike) -> None:
        arr = np.asanyarray(data)
        if arr.ndim != 3:
            raise Exception("Volumes must be 3-dimensional")
        # Coerce the data to something that can be displayed by pygfx
        processed = _coerce_data(arr, n_spatial=3)

        # If we have a texture already, see whether we can reuse it:
        current: pygfx.Texture | None = getattr(self, "_texture", None)
        if current is not None and _can_reuse_volume(current, processed):
            # To reuse, we overwrite its buffer and mark dirty.
            current.data[:] = processed  # pyright: ignore
            current.update_full()
        else:
            # Copy so later in-place reuse won't mutate the originally-passed array.
            new_buffer = processed.copy()
            self._texture = pygfx.Texture(new_buffer, dim=new_buffer.ndim)
            self._pygfx_node.geometry = pygfx.Geometry(grid=self._texture)

        # Keep transform compensation in sync whenever data or factors change.
        self._snx_set_transform(self._model.transform)

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


def _can_reuse_volume(texture: pygfx.Texture, processed: np.ndarray) -> bool:
    """True if *texture* can hold *processed* without reallocation."""
    return (
        texture.data is not None
        and texture.data.shape == processed.shape
        and texture.data.dtype == processed.dtype
    )
