from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pygfx

from scenex.adaptors._base import CameraAdaptor

from ._node import Node

if TYPE_CHECKING:
    from scenex import model

logger = logging.getLogger("scenex.adaptors.pygfx")


class Camera(Node, CameraAdaptor):
    """Adaptor for pygfx camera."""

    _pygfx_node: pygfx.Camera
    pygfx_controller: pygfx.Controller

    def __init__(self, camera: model.Camera, **backend_kwargs: Any) -> None:
        self._camera_model = camera
        self._pygfx_node = pygfx.Camera()

        self._pygfx_node.local.scale_y = -1  # don't think this is working...

    def _snx_set_type(self, arg: model.CameraType) -> None:
        logger.warning("'Camera._snx_set_type' not implemented for pygfx")

    def _view_size(self) -> tuple[float, float] | None:
        """Return the size of first parent viewbox in pixels."""
        logger.warning("'Camera._view_size' not implemented for pygfx")
        return None

    def _snx_set_projection(self, arg: model.Transform) -> None:
        self._pygfx_node.projection_matrix = arg.root  # pyright: ignore[reportAttributeAccessIssue]

    def _snx_set_controller(self, arg: model.InteractionStrategy | None) -> None:
        pass
