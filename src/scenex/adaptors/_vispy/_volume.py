from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import vispy.scene
import vispy.visuals

from scenex.adaptors._base import VolumeAdaptor

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap
    from numpy.typing import ArrayLike

    from scenex import model


class Volume(Node, VolumeAdaptor):
    """vispy backend adaptor for a Volume node."""

    _vispy_node: vispy.visuals.VolumeVisual

    def __init__(self, volume: model.Volume, **backend_kwargs: Any) -> None:
        # TODO: What if volume.data is None?
        self._vispy_node = vispy.scene.Volume(
            volume.data, texture_format="auto", **backend_kwargs
        )

    def _snx_set_cmap(self, arg: Colormap) -> None:
        self._vispy_node.cmap = arg.to_vispy()

    def _snx_set_clims(self, arg: tuple[float, float] | None) -> None:
        self._vispy_node.clim = arg

    def _snx_set_gamma(self, arg: float) -> None:
        self._vispy_node.gamma = arg

    def _snx_set_interpolation(self, arg: model.InterpolationMode) -> None:
        self._vispy_node.interpolation = arg

    def _snx_set_data(self, data: ArrayLike) -> None:
        self._vispy_node.set_data(np.asarray(data))

    def _snx_set_render_mode(
        self,
        data: model.RenderMode,
        interpolation: model.InterpolationMode | None = None,
    ) -> None:
        self._vispy_node.method = data
