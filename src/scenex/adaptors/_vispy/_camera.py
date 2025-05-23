from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import vispy.scene

from scenex.adaptors._base import CameraAdaptor

from ._node import Node

if TYPE_CHECKING:
    from scenex import model
    from scenex.model import Transform


class Camera(Node, CameraAdaptor):
    """Adaptor for pygfx camera."""

    _vispy_node: vispy.scene.BaseCamera

    def __init__(self, camera: model.Camera, **backend_kwargs: Any) -> None:
        self._camera_model = camera
        if camera.type == "panzoom":
            self._vispy_node = vispy.scene.PanZoomCamera()
            self._vispy_node.interactive = True
        elif camera.type == "perspective":
            # TODO: These settings were copied from the pygfx camera.
            # Unify these values?
            self._vispy_node = vispy.scene.ArcballCamera(70)

        self._snx_zoom_to_fit(0.1)

    def _snx_set_zoom(self, zoom: float) -> None:
        self._vispy_node.zoom_factor = zoom

    def _snx_set_center(self, arg: tuple[float, ...]) -> None:
        self._vispy_node.center = arg

    def _snx_set_type(self, arg: model.CameraType) -> None:
        raise NotImplementedError()

    def _snx_set_transform(self, arg: Transform) -> None:
        if isinstance(self._vispy_node, vispy.scene.PanZoomCamera):
            self._vispy_node.tf_mat = vispy.scene.transforms.MatrixTransform(
                np.asarray(arg)
            )
        else:
            super()._snx_set_transform(arg)

    def _view_size(self) -> tuple[float, float] | None:
        """Return the size of first parent viewbox in pixels."""
        raise NotImplementedError

    def _snx_zoom_to_fit(self, margin: float) -> None:
        # reset camera to fit all objects
        self._vispy_node.set_range()
