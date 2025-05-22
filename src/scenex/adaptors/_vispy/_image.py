from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vispy.scene
import vispy.visuals

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap
    from numpy.typing import ArrayLike

    from scenex import model


class Image(Node):
    """pygfx backend adaptor for an Image node."""

    _vispy_node: vispy.visuals.ImageVisual

    def __init__(self, image: model.Image, **backend_kwargs: Any) -> None:
        self._vispy_node = vispy.scene.Image(
            data=image.data, texture_format="auto", **backend_kwargs
        )
        self._snx_set_data(image.data)
        self._vispy_node.visible = True

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
