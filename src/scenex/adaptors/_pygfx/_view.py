from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pygfx

from scenex.adaptors._base import ViewAdaptor

from ._adaptor_registry import get_adaptor

if TYPE_CHECKING:
    from cmap import Color

    from scenex import model

    from . import _camera, _scene

logger = logging.getLogger("scenex.adaptors.pygfx")


class View(ViewAdaptor):
    """View interface for pygfx Backend.

    A view combines a scene and a camera to render a scene (onto a canvas).
    """

    _pygfx_scene: pygfx.Scene
    _pygfx_cam: pygfx.Camera

    def __init__(self, view: model.View, **backend_kwargs: Any) -> None:
        self._model = view

        self._snx_set_scene(view.scene)
        self._snx_set_camera(view.camera)

        # -- Layout components -- #
        self._background_mat = pygfx.BackgroundMaterial()
        self._background = pygfx.Background(None, material=self._background_mat)
        self._pygfx_scene.add(self._background)

        # -- Layout connections -- #
        self._model.layout.events.background_color.connect(self._set_background_color)

        # -- Layout initialization -- #
        self._set_background_color(view.layout.background_color)

    def _snx_set_visible(self, arg: bool) -> None:
        pass

    def _snx_set_scene(self, scene: model.Scene) -> None:
        self._scene_adaptor = cast("_scene.Scene", get_adaptor(scene))
        self._pygfx_scene = self._scene_adaptor._pygfx_node

    def _snx_set_camera(self, cam: model.Camera) -> None:
        self._cam_adaptor = cast("_camera.Camera", get_adaptor(cam))
        self._pygfx_cam = self._cam_adaptor._pygfx_node

    def _draw(self, renderer: pygfx.renderers.WgpuRenderer) -> None:
        rect = self._model.layout.content_rect
        # FIXME: On Qt, for HiDPI screens, the logical screen size (the rect
        # variable above) can, through rounding error during resizing, become
        # slightly larger than the physical size, which causes pygfx to error.
        # This code "fixes" it but I think we could do better...maybe upstream?
        ratio = renderer.physical_size[1] / renderer.logical_size[1]  # pyright:ignore
        if (rect[0] + rect[2]) * ratio > renderer.physical_size[0]:
            # content rect is too wide for the canvas - adjust width
            new_width = int(renderer.physical_size[0] / ratio - rect[0])
            rect = (rect[0], rect[1], new_width, rect[3])
        if (rect[1] + rect[3]) * ratio > renderer.physical_size[1]:
            # content rect is too tall for the canvas - adjust height
            new_height = int(renderer.physical_size[1] / ratio - rect[1])
            rect = (rect[0], rect[1], rect[2], new_height)
        # End FIXME

        renderer.render(self._pygfx_scene, self._pygfx_cam, rect=rect, flush=False)

    def _set_background_color(self, color: Color | None) -> None:
        rgba = color.rgba if color is not None else (0, 0, 0, 0)
        self._background_mat.set_colors(rgba)

    def _snx_render(self) -> np.ndarray:
        """Render to offscreen buffer."""
        from rendercanvas.offscreen import OffscreenRenderCanvas

        canvas = OffscreenRenderCanvas(size=(640, 480), pixel_ratio=2)
        renderer = pygfx.renderers.WgpuRenderer(canvas)

        canvas.request_draw(lambda: renderer.render(self._pygfx_scene, self._pygfx_cam))
        return np.asarray(canvas.draw())
