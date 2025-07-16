from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vispy.geometry
import vispy.scene

from scenex.adaptors._base import CameraAdaptor
from scenex.model import Transform

from ._node import Node

if TYPE_CHECKING:
    from scenex import model


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
        # 1) A transform from world space to local space:
        #   Note that this is usually the inverse of the model's transform matrix
        self._transform = Transform()
        # 2) A transform from local space to NDC:
        self._projection = Transform()
        # 3) A transform from NDC to canvas position:
        self._from_NDC = Transform()

        self._vispy_node = vispy.scene.BaseCamera()
        # FIXME: Compared to pygfx, the y-axis appears inverted.
        # The line below does not help...
        # self._vispy_node.flip = (False, True, False)

    def _set_view(self, view: vispy.scene.ViewBox) -> None:
        # map [-1, -1] to [0, 0]
        # map [1, 1] to [w, h]
        w, h = view.size
        self._from_NDC = Transform().translated((1, 1)).scaled((w / 2, h / 2, 1))

        self._update_vispy_node_tform()

    def _snx_set_type(self, arg: model.CameraType) -> None:
        raise NotImplementedError()

    def _snx_set_transform(self, arg: Transform) -> None:
        # Note that the scenex transform is inverted here.
        # Scenex transforms map local coordinates into parent coordinates,
        # but our vispy node's transform must go the opposite way, from world
        # coordinates into parent coordinates.
        #
        # FIXME: Note the discrepancy between world and parent coordinates. World
        # coordinates are needed for the vispy transform node, but the current transform
        # only converts local to parent space. This will likely be a source of bugs for
        # more complicated scenes. There's also a TODO above about fixing this.
        self._transform = arg.inv()
        self._update_vispy_node_tform()

    def _snx_set_projection(self, arg: Transform) -> None:
        self._projection = arg
        # Have to recompute the vispy transform offset if the projection changed
        self._snx_set_transform(self._camera_model.transform)
        # FIXME this call is redundant since _snx_set_transform does it, but it's
        # worth remembering that this needs to happen.
        self._update_vispy_node_tform()

    def _update_vispy_node_tform(self) -> None:
        mat = self._transform @ self._projection.T @ self._from_NDC
        self._vispy_node.transform = vispy.scene.transforms.MatrixTransform(mat.root)
        self._vispy_node.view_changed()

    def _view_size(self) -> tuple[float, float] | None:
        """Return the size of first parent viewbox in pixels."""
        raise NotImplementedError

    def _snx_zoom_to_fit(self, margin: float) -> None:
        # reset camera to fit all objects
        # FIXME: Implement this code in the model
        self._vispy_node.set_range(margin=margin)
