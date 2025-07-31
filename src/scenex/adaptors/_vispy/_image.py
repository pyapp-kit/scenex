from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vispy.scene
import vispy.visuals

from scenex.adaptors._base import ImageAdaptor

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap
    from numpy.typing import ArrayLike

    from scenex import model
    from scenex.model._transform import Transform


class Image(Node, ImageAdaptor):
    """pygfx backend adaptor for an Image node."""

    _vispy_node: vispy.visuals.ImageVisual

    def __init__(self, image: model.Image, **backend_kwargs: Any) -> None:
        self._vispy_node = vispy.scene.Image(
            data=image.data, texture_format="auto", **backend_kwargs
        )
        self._snx_set_data(image.data)
        self._vispy_node.visible = True
        self._vispy_node.interactive = True

    def _snx_set_transform(self, arg: Transform) -> None:
        # Offset accounting for vispy's pixel centers at half-integer locations
        offset = arg.map([-0.5, -0.5, 0, 0])
        super()._snx_set_transform(arg.translated(offset))

    def _snx_set_cmap(self, arg: Colormap) -> None:
        self._vispy_node.cmap = arg.to_vispy()

    def _snx_set_clims(self, arg: tuple[float, float] | None) -> None:
        self._vispy_node.clim = arg

    def _snx_set_gamma(self, arg: float) -> None:
        self._vispy_node.gamma = arg

    def _snx_set_interpolation(self, arg: model.InterpolationMode) -> None:
        self._vispy_node.interpolation = arg

    def _snx_set_data(self, data: ArrayLike) -> None:
        self._vispy_node.set_data(data)
