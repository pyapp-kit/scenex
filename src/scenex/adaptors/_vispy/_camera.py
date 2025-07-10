from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import vispy.geometry
import vispy.scene

from scenex.adaptors._base import CameraAdaptor
from scenex.model import Transform

from ._node import Node

if TYPE_CHECKING:
    from scenex import model


class _Arcball(vispy.scene.ArcballCamera):
    def _get_dim_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return np.array((0, +1, 0)), np.array((0, 0, +1)), np.array((+1, 0, 0))
        # # Specify up and forward vector
        # M = {'+z': [(0, 0, +1), (0, 1, 0)],
        #      '-z': [(0, 0, -1), (0, 1, 0)],
        #      '+y': [(0, +1, 0), (1, 0, 0)],
        #      '-y': [(0, -1, 0), (1, 0, 0)],
        #      '+x': [(+1, 0, 0), (0, 0, 1)],
        #      '-x': [(-1, 0, 0), (0, 0, 1)],
        #      }
        # up, forward = M[self.up]
        # right = np.cross(forward, up)
        # return np.array(up), np.array(forward), right


class Camera(Node, CameraAdaptor):
    """Adaptor for pygfx camera."""

    _vispy_node: vispy.scene.BaseCamera

    def __init__(self, camera: model.Camera, **backend_kwargs: Any) -> None:
        self._camera_model = camera
        if camera.type == "panzoom":
            self._vispy_node = vispy.scene.PanZoomCamera()
            self._vispy_node.flip = (False, True, False)
            self._vispy_node.interactive = True
        elif camera.type == "perspective":
            # TODO: These settings were copied from the pygfx camera.
            # Unify these values?
            self._vispy_node = _Arcball(70)
            self._vispy_node.up = "+y"

        self._snx_zoom_to_fit(0.1)

    def _snx_set_zoom(self, zoom: float) -> None:
        self._vispy_node.zoom_factor = zoom

    def _snx_set_center(self, arg: tuple[float, ...]) -> None:
        self._vispy_node.center = arg

    def _snx_set_type(self, arg: model.CameraType) -> None:
        raise NotImplementedError()

    def _snx_set_transform(self, arg: Transform) -> None:
        # FIXME: Handle scaling
        # FIXME: Y-panning inverted?
        self._vispy_node.center = tuple(arg.root[3, :3])

    def _snx_set_projection(self, arg: Transform) -> None:
        pass
        # self._pygfx_node.projection_matrix = arg.root

    def _view_size(self) -> tuple[float, float] | None:
        """Return the size of first parent viewbox in pixels."""
        raise NotImplementedError

    def _snx_zoom_to_fit(self, margin: float) -> None:
        # reset camera to fit all objects
        self._vispy_node.set_range(margin=margin)
        vis_tform = self._vispy_node.transform

        tform = Transform()
        if isinstance(vis_tform, vispy.scene.transforms.STTransform):
            vis_matrix = cast(
                "vispy.scene.transforms.MatrixTransform", vis_tform.as_matrix()
            )
            tform = Transform(vis_matrix.matrix)
        elif isinstance(vis_tform, vispy.scene.transforms.MatrixTransform):
            tform = Transform(vis_tform.matrix)

        # Vispy's camera transforms map canvas coordinates to world coordinates.
        # Thus the projection matrix should map NDC coordinates to canvas
        # coordinates, to obtain the desired effect of mapping NDC coordinates in
        # scenex to world coordinates through the projection and transform matrices.
        if vb := self._vispy_node.viewbox:
            w, h = cast("tuple[float, float]", vb.size)
            # This transform maps NDC coordinates to canvas position
            self._de_NDC = (
                Transform().translated((-w / 2, -h / 2)).scaled((2 / w, 2 / h, 1)).T
            )
            # This transform maps NDC coordinates to TRANSFORMED world coordinates
            tform = self._de_NDC @ tform.T

        untranslated_tform = tform.root.copy()
        untranslated_tform[:3, 3] = 0.0
        self._camera_model.projection = Transform(untranslated_tform)

        self._camera_model.transform = Transform().translated(self._vispy_node.center)
