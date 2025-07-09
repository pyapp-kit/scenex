from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap
    from numpy.typing import ArrayLike

    from scenex import model
    from scenex.model._transform import Transform

logger = logging.getLogger("scenex.adaptors.pygfx")


class Image(Node):
    """pygfx backend adaptor for an Image node."""

    _pygfx_node: pygfx.Image
    _material: pygfx.ImageBasicMaterial

    def __init__(self, image: model.Image, **backend_kwargs: Any) -> None:
        self._material = pygfx.ImageBasicMaterial(clim=image.clims)
        self._pygfx_node = pygfx.Image(None, self._material)
        self._model = image
        self._snx_set_data(image.data)

    def _snx_set_cmap(self, arg: Colormap) -> None:
        self._material.map = arg.to_pygfx()

    def _snx_set_clims(self, arg: tuple[float, float] | None) -> None:
        self._material.clim = arg

    def _snx_set_gamma(self, arg: float) -> None:
        self._material.gamma = arg

    def _snx_set_transform(self, arg: Transform) -> None:
        # Offset due to locating the center of pixels at integer locations.
        super()._snx_set_transform(arg.translated([0.5, 0.5]))

    def _snx_set_interpolation(self, arg: model.InterpolationMode) -> None:
        if arg == "bicubic":
            logger.warning(
                "Bicubic interpolation not supported by pygfx - falling back to linear",
                RuntimeWarning,
                stacklevel=2,
            )
            self._model.interpolation = "linear"
            return
        self._material.interpolation = arg

    def _create_texture(self, data: ArrayLike | None) -> pygfx.Texture:
        data = np.asanyarray(data)
        if data is not None:
            dim = data.ndim
            if dim > 2 and data.shape[-1] <= 4:
                dim -= 1  # last array dim is probably (a subset of) rgba
        else:
            dim = 2
        # TODO: unclear whether get_view() is better here...
        return pygfx.Texture(data, dim=dim)

    def _snx_set_data(self, data: ArrayLike) -> None:
        self._texture = self._create_texture(data)
        self._pygfx_node.geometry = pygfx.Geometry(grid=self._texture)
