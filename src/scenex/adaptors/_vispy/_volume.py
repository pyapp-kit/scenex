from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vispy.scene
import vispy.visuals

from scenex.adaptors._base import VolumeAdaptor
from scenex.adaptors._vispy._image import _coerce_data

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap
    from numpy.typing import ArrayLike

    from scenex import model
    from scenex.model._transform import Transform


class Volume(Node, VolumeAdaptor):
    """vispy backend adaptor for a Volume node."""

    _vispy_node: vispy.visuals.VolumeVisual

    def __init__(self, volume: model.Volume, **backend_kwargs: Any) -> None:
        self._model = volume
        # TODO: What if volume.data is None?
        self._vispy_node = vispy.scene.Volume(
            _coerce_data(volume.data, n_spatial=3),
            texture_format="auto",
            **backend_kwargs,
        )

    def _snx_set_transform(self, arg: Transform) -> None:
        # Offset accounting for vispy's pixel centers at half-integer locations
        offset = arg.map([-0.5, -0.5, -0.5, 0])
        # Compensate for downscaled textures
        # Note that volume axes are ZYX
        x_fac = self._model.data.shape[2] / self._vispy_node._texture.shape[2]  # pyright: ignore
        y_fac = self._model.data.shape[1] / self._vispy_node._texture.shape[1]  # pyright: ignore
        z_fac = self._model.data.shape[0] / self._vispy_node._texture.shape[0]  # pyright: ignore
        super()._snx_set_transform(arg.scaled((x_fac, y_fac, z_fac)).translated(offset))

    def _snx_set_cmap(self, arg: Colormap) -> None:
        self._vispy_node.cmap = arg.to_vispy()

    def _snx_set_clims(self, arg: tuple[float, float] | None) -> None:
        self._vispy_node.clim = arg

    def _snx_set_gamma(self, arg: float) -> None:
        self._vispy_node.gamma = arg

    def _snx_set_interpolation(self, arg: model.InterpolationMode) -> None:
        self._vispy_node.interpolation = arg

    def _snx_set_data(self, data: ArrayLike) -> None:
        # Coerce the data to something that can be displayed by vispy
        processed = _coerce_data(data, n_spatial=3)
        self._vispy_node.set_data(processed)

    def _snx_set_render_mode(
        self,
        data: model.RenderMode,
        interpolation: model.InterpolationMode | None = None,
    ) -> None:
        self._vispy_node.method = data
