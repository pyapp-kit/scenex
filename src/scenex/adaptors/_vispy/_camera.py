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

        # The camera model contains:
        # A projection transform, mapping local space to NDC
        # A parent transform, mapping parent space to local space
        #
        # TODO: We may need a utility to get a transform mapping world space to local
        # space.
        #
        # The BaseCamera.transform field should map world space to canvas position.
        #
        # To construct this transform from our camera model, we need:
        # 1) A transform from world space to local space (self._camera_model.transform)
        self._transform = Transform()
        # 2) A transform from local space to NDC (self._camera_model.projection)
        self._projection = Transform()
        # 3) A transform from NDC to canvas position:
        self._from_NDC = Transform()

        if camera.type == "panzoom":
            self._vispy_node = vispy.scene.BaseCamera()
            self._vispy_node.flip = (False, True, False)
            # self._vispy_node.interactive = True
        elif camera.type == "perspective":
            # TODO: These settings were copied from the pygfx camera.
            # Unify these values?
            self._vispy_node = _Arcball(70)
            self._vispy_node.up = "+y"

        self._snx_zoom_to_fit(0.1)

    def _set_view(self, view: vispy.scene.ViewBox) -> None:
        # map [-1, -1] to [0, 0]
        # map [1, 1] to [w, h]
        w, h = view.size
        self._from_NDC = Transform().translated((1, 1)).scaled((w / 2, h / 2, 1))
        # TODO: Delete
        # cam = vispy.scene.PanZoomCamera()
        # cam.flip = [False, True, False]
        # v = vispy.scene.ViewBox(cam)
        # c = cam.transform.as_matrix()
        # t = Transform().translated((-0.5, -0.5))
        # p = projections.orthographic(1, 1, 2_000_000)
        # c_rep = t @ p @ self._from_NDC
        self._update_vispy_node_tform()
        return None

    def _snx_set_type(self, arg: model.CameraType) -> None:
        raise NotImplementedError()

    def _snx_set_transform(self, arg: Transform) -> None:
        # The vispy camera's transformation matrix maps [0, 0] to the top left corner of
        # the camera. Since the model transform maps [0, 0] to the CENTER of the camera,
        # we have to offset the transform
        # offset_mat = self._camera_model.projection
        # offset = offset_mat.imap((-1, -1)) - offset_mat.imap((0, 0))
        self._transform = arg.inv()
        self._update_vispy_node_tform()
        # FIXME: Handle scaling
        # FIXME: Y-panning inverted?
        # self._vispy_node.center = tuple(arg.root[3, :3])

    def _snx_set_projection(self, arg: Transform) -> None:
        self._projection = arg
        # Have to recompute the vispy transform offset if the projection changed
        self._snx_set_transform(self._camera_model.transform)
        # FIXME this call is redundant since _snx_set_transform does it, but it's
        # worth remembering that this needs to happen.
        self._update_vispy_node_tform()

    def _update_vispy_node_tform(self) -> None:
        mat = self._transform @ self._projection @ self._from_NDC
        self._vispy_node.transform = vispy.scene.transforms.MatrixTransform(mat.root)
        self._vispy_node.view_changed()

    def _view_size(self) -> tuple[float, float] | None:
        """Return the size of first parent viewbox in pixels."""
        raise NotImplementedError

    def _snx_zoom_to_fit(self, margin: float) -> None:
        # reset camera to fit all objects
        # FIXME: Implement this code in the model
        self._vispy_node.set_range(margin=margin)
        return
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
            # This transform maps NDC coordinates to TRANSFORMED world coordinates
            self._de_NDC = ()
            tform = self._from_NDC.T @ tform.T

        untranslated_tform = tform.root.copy()
        untranslated_tform[:3, 3] = 0.0
        self._camera_model.projection = Transform(untranslated_tform)

        self._camera_model.transform = Transform().translated(self._vispy_node.center)
        return
