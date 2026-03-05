from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import vispy
import vispy.app
import vispy.scene
import vispy.scene.subscene
from vispy.geometry import Rect

from scenex.adaptors._base import ViewAdaptor

from ._adaptor_registry import get_adaptor

if TYPE_CHECKING:
    from cmap import Color

    from scenex import model

    from . import _camera, _scene


class View(ViewAdaptor):
    """View interface for vispy Backend.

    A view combines a scene and a camera to render a scene (onto a canvas).
    """

    _vispy_scene: vispy.scene.subscene.SubScene
    _vispy_viewbox: vispy.scene.ViewBox
    _vispy_camera: vispy.scene.BaseCamera

    def __init__(self, view: model.View, **backend_kwargs: Any) -> None:
        self._model = view
        self._vispy_viewbox = vispy.scene.ViewBox()
        self._vispy_viewbox.events.resize.connect(self._on_vispy_viewbox_resized)  # pyright: ignore

        self._snx_set_camera(view.camera)
        self._snx_set_scene(view.scene)

        # FIXME: Should also listen for connection to a view, and similarly set the rect
        # -- Layout connections -- #
        self._model.layout.events.background_color.connect(self._set_background_color)
        view.layout.events.all.connect(self._on_layout_changed)

        # -- Layout initialization -- #
        self._set_background_color(view.layout.background_color)
        self._on_layout_changed()

    def _on_vispy_viewbox_resized(self, event: Any) -> None:
        # Update camera's _from_NDC transform
        w, h = self._vispy_viewbox.rect.size
        self._cam_adaptor._set_view(w, h)

    def _on_layout_changed(self, event: Any | None = None) -> None:
        if canvas := self._model.canvas:
            rect = Rect(canvas.content_rect_for(self._model))
            self._vispy_viewbox.rect = rect
            self._vispy_viewbox.update()
            self._cam_adaptor._set_view(rect.width, rect.height)

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

        # Add the camera to the scene
        if hasattr(self, "_vispy_cam"):
            self._vispy_camera.parent = new
        self._vispy_scene = new

    def _snx_set_camera(self, cam: model.Camera) -> None:
        self._cam_adaptor = cast("_camera.Camera", get_adaptor(cam))
        self._vispy_camera = self._cam_adaptor._vispy_node
        if hasattr(self, "_vispy_viewbox"):
            self._vispy_viewbox.camera = self._vispy_camera
            # Vispy camera transforms need knowledge of viewbox
            # (specifically, its size)
            self._vispy_viewbox.update()
            self._cam_adaptor._set_view(*self._vispy_viewbox.size)

    def _draw(self) -> None:
        self._vispy_viewbox.update()

    def _set_background_color(self, color: Color | None) -> None:
        color_data = None if color is None else color.rgba
        self._vispy_viewbox.bgcolor = color_data

    def _snx_render(self) -> np.ndarray:
        """Render to screenshot."""
        canvas = cast("vispy.scene.Node", self._vispy_viewbox).canvas
        if isinstance(canvas, vispy.app.Canvas):
            return np.asarray(canvas.render())
        return np.zeros((640, 480))
