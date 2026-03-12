from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx

from scenex.adaptors._base import ImageAdaptor

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap
    from numpy.typing import ArrayLike

    from scenex import model

logger = logging.getLogger("scenex.adaptors.pygfx")

# Certain numpy data types are not supported by pygfx. We downcast them to another type
# defined in this dict.
DOWNCASTS = {
    np.dtype("float64"): np.dtype("float32"),
    np.dtype("int64"): np.dtype("int32"),
    np.dtype("uint64"): np.dtype("uint32"),
}


class Image(Node, ImageAdaptor):
    """pygfx backend adaptor for an Image node."""

    _pygfx_node: pygfx.Image
    _material: pygfx.ImageBasicMaterial
    _geometry: pygfx.Geometry

    def __init__(self, image: model.Image, **backend_kwargs: Any) -> None:
        self._model = image
        self._material = pygfx.ImageBasicMaterial(
            clim=image.clims,
            # This value has model render order win for coplanar objects
            depth_compare="<=",
        )
        self._snx_set_data(image.data)
        self._pygfx_node = pygfx.Image(self._geometry, self._material)

    def _snx_set_cmap(self, arg: Colormap) -> None:
        if np.asarray(self._model.data).ndim == 3:
            self._material.map = None
        else:
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
            self._model.interpolation = "linear"
            return
        self._material.interpolation = arg

    def _create_texture(self, data: ArrayLike | None) -> pygfx.Texture:
        if data is not None:
            data = np.asanyarray(data)
            dim = data.ndim
            if dim > 2 and data.shape[-1] <= 4:
                dim -= 1  # last array dim is probably (a subset of) rgba
            if data.dtype in DOWNCASTS:
                cast_to = DOWNCASTS[data.dtype]
                # pygfx doesn't support 64-bit dtypes; downcast transparently.
                # The user hasn't done anything wrong — this is a backend limitation.
                logger.warning(
                    "Downcasting image data from %s to %s for pygfx compatibility",
                    data.dtype.name,
                    cast_to.name,
                )
                data = data.astype(cast_to)
        else:
            dim = 2
        # TODO: unclear whether get_view() is better here...
        return pygfx.Texture(data, dim=dim)

    def _snx_set_data(self, data: ArrayLike) -> None:
        self._texture = self._create_texture(data)
        self._geometry = pygfx.Geometry(grid=self._texture)
        if hasattr(self, "_pygfx_node"):
            self._pygfx_node.geometry = self._geometry
        if np.asarray(data).ndim == 3:
            self._material.map = None
        else:
            self._material.map = self._model.cmap.to_pygfx()
