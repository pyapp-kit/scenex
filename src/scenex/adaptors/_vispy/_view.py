from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import vispy
import vispy.app
import vispy.color
import vispy.scene
import vispy.scene.subscene

from scenex.adaptors._base import ViewAdaptor

from ._adaptor_registry import get_adaptor

if TYPE_CHECKING:
    from cmap import Color

    from scenex import model

    from . import _camera, _scene


BLENDING_MAP = {
    "default": "default",
    "opaque": "opaque",
    "alpha": "ordered1",
    "additive": "additive",
}


class View(ViewAdaptor):
    """View interface for vispy Backend.

    A view combines a scene and a camera to render a scene (onto a canvas).
    """

    _vispy_scene: vispy.scene.subscene.SubScene
    _vispy_viewbox: vispy.scene.ViewBox
    _vispy_camera: vispy.scene.BaseCamera

    def __init__(self, view: model.View, **backend_kwargs: Any) -> None:
        self._vispy_viewbox = vispy.scene.ViewBox()

        self._snx_set_camera(view.camera)
        self._snx_set_scene(view.scene)
        self._snx_set_blending(view.blending)

    def _snx_get_native(self) -> Any:
        return self._vispy_viewbox

    def _snx_set_blending(self, arg: model.BlendMode) -> None:
        # FIXME: This is tricky - needs to be passed to set_gl_state for each scene
        # node.
        pass

    def _snx_set_visible(self, arg: bool) -> None:
        pass

    def _snx_set_scene(self, scene: model.Scene) -> None:
        # Remove the old scene from the viewbox
        prev = self._vispy_viewbox.scene
        if prev is not None:
            cast("vispy.scene.subscene.SubScene", prev).parent = None

        # Set the scene through a private attribute
        # Unfortunate there's no public attr for this.
        new = cast("_scene.Scene", get_adaptor(scene))._vispy_node
        self._vispy_viewbox._scene = new
        new.parent = self._vispy_viewbox

        # FIXME: Viewbox expects there to always be a Clipper
        # Let's just copy the old one?
        if isinstance(prev, vispy.scene.subscene.SubScene):
            new._clipper = prev.clipper
            new.clip_children = prev.clip_children

        # Add the camera to the scene
        if hasattr(self, "_vispy_cam"):
            self._vispy_camera.parent = new

    def _snx_set_camera(self, cam: model.Camera) -> None:
        self._cam_adaptor = cast("_camera.Camera", get_adaptor(cam))
        self._vispy_camera = self._cam_adaptor._vispy_node
        if hasattr(self, "_vispy_viewbox"):
            self._vispy_viewbox.camera = self._vispy_camera

    def _draw(self) -> None:
        self._vispy_viewbox.update()

    def _snx_set_position(self, arg: tuple[float, float]) -> None:
        raise NotImplementedError()

    def _snx_set_size(self, arg: tuple[float, float] | None) -> None:
        raise NotImplementedError()

    def _snx_set_border_width(self, arg: float) -> None:
        warnings.warn(
            "set_border_width not implemented for vispy", RuntimeWarning, stacklevel=2
        )

    def _snx_set_border_color(self, arg: Color | None) -> None:
        warnings.warn(
            "set_border_color not implemented for vispy", RuntimeWarning, stacklevel=2
        )

    def _snx_set_padding(self, arg: int) -> None:
        warnings.warn(
            "set_padding not implemented for vispy", RuntimeWarning, stacklevel=2
        )

    def _snx_set_margin(self, arg: int) -> None:
        warnings.warn(
            "set_margin not implemented for vispy", RuntimeWarning, stacklevel=2
        )

    def _snx_render(self) -> np.ndarray:
        """Render to screenshot."""
        canvas = cast("vispy.scene.Node", self._vispy_viewbox).canvas
        if isinstance(canvas, vispy.app.Canvas):
            return np.asarray(canvas.render())
        return np.zeros((640, 480))
