from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pygfx

from scenex.adaptors._base import CameraAdaptor

from ._node import Node

if TYPE_CHECKING:
    from scenex import model
    from scenex.model import Transform

logger = logging.getLogger("scenex.adaptors.pygfx")


class Camera(Node, CameraAdaptor):
    """Adaptor for pygfx camera."""

    _pygfx_node: pygfx.PerspectiveCamera
    pygfx_controller: pygfx.Controller

    def __init__(self, camera: model.Camera, **backend_kwargs: Any) -> None:
        self._camera_model = camera
        # FIXME: This won't always hold as the projection matrix changes.
        # Once we have better controllers via event filters, the pygfx_controller
        # field should disappear and the _pygfx_node should just be a pygfx.Camera.
        self._pygfx_node = pygfx.OrthographicCamera()
        self.pygfx_controller = pygfx.PanZoomController(self._pygfx_node)

        self._pygfx_node.local.scale_y = -1  # don't think this is working...

    def _snx_set_type(self, arg: model.CameraType) -> None:
        logger.warning("'Camera._snx_set_type' not implemented for pygfx")

    def _view_size(self) -> tuple[float, float] | None:
        """Return the size of first parent viewbox in pixels."""
        logger.warning("'Camera._view_size' not implemented for pygfx")
        return None

    def update_controller(self) -> None:
        # This is called by the View Adaptor in the `_visit` method
        # ... which is in turn called by the Canvas backend adaptor's `_animate` method
        # i.e. the main render loop.
        self.pygfx_controller.update_camera(self._pygfx_node)

    def set_viewport(self, viewport: pygfx.Viewport) -> None:
        # This is used by the Canvas backend adaptor...
        # and should perhaps be moved to the View Adaptor
        self.pygfx_controller.add_default_event_handlers(viewport, self._pygfx_node)

    def _snx_set_projection(self, arg: Transform) -> None:
        self._pygfx_node.projection_matrix = arg.root  # pyright: ignore[reportAttributeAccessIssue]
