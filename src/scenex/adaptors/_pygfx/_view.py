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
        self._renderer: pygfx.renderers.WgpuRenderer | None = None

        self._snx_set_scene(view.scene)
        self._snx_set_camera(view.camera)
        # TODO: this is needed... but breaks tests until we deal with Layout better.
        # self._snx_set_background_color(view.layout.background_color)

    def _set_pygfx_canvas(self, canvas: Any, x: int, y: int) -> None:
        self._renderer = pygfx.renderers.WgpuRenderer(canvas)

    def _snx_get_native(self) -> pygfx.Viewport:
        return pygfx.Viewport(self._renderer)

    def _snx_set_visible(self, arg: bool) -> None:
        pass

    def _snx_set_scene(self, scene: model.Scene) -> None:
        self._scene_adaptor = cast("_scene.Scene", get_adaptor(scene))
        self._pygfx_scene = self._scene_adaptor._pygfx_node

    def _snx_set_camera(self, cam: model.Camera) -> None:
        self._cam_adaptor = cast("_camera.Camera", get_adaptor(cam))
        self._pygfx_cam = self._cam_adaptor._pygfx_node

    def _draw(self) -> None:
        if self._renderer:
            rect = self._model.layout.content_rect
            # FIXME: On Qt, for HiDPI screens, the logical screen size (the rect
            # variable above) can, through rounding error during resizing, become
            # slightly larger than the physical size, which causes pygfx to error.
            # This code "fixes" it but I think we could do better...maybe upstream?
            ratio = self._renderer.physical_size[1] / self._renderer.logical_size[1]  # pyright:ignore
            if rect[2] * ratio > self._renderer.physical_size[0]:
                # content rect is too wide for the canvas - adjust width
                new_width = int(self._renderer.physical_size[0] / ratio)
                rect = (rect[0], rect[1], new_width, rect[3])
            if rect[3] * ratio > self._renderer.physical_size[1]:
                # content rect is too tall for the canvas - adjust height
                new_height = int(self._renderer.physical_size[1] / ratio)
                rect = (rect[0], rect[1], rect[2], new_height)
            # End FIXME

            self._renderer.render(self._pygfx_scene, self._pygfx_cam, rect=rect)
            self._renderer.request_draw()

    def _snx_set_position(self, arg: tuple[float, float]) -> None:
        logger.warning("View.set_position not implemented for pygfx")

    def _snx_set_size(self, arg: tuple[float, float] | None) -> None:
        if arg is None:
            logger.warning(
                "Ignoring View.set_size(None): Don't know how to handle this..."
            )
        else:
            r = self._snx_get_native().rect
            self._snx_get_native().rect = (r[0], r[1], arg[0], arg[1])
            # FIXME: Camera projection transform should also be updated...

    def _snx_set_background_color(self, color: Color | None) -> None:
        colors = (color.rgba,) if color is not None else ()
        background = pygfx.Background(None, material=pygfx.BackgroundMaterial(*colors))
        self._pygfx_scene.add(background)

    def _snx_set_border_width(self, arg: float) -> None:
        logger.warning("View.set_border_width not implemented for pygfx")

    def _snx_set_border_color(self, arg: Color | None) -> None:
        logger.warning("View.set_border_color not implemented for pygfx")

    def _snx_set_padding(self, arg: int) -> None:
        logger.warning("View.set_padding not implemented for pygfx")

    def _snx_set_margin(self, arg: int) -> None:
        logger.warning("View.set_margin not implemented for pygfx")

    def _snx_render(self) -> np.ndarray:
        """Render to offscreen buffer."""
        from rendercanvas.offscreen import OffscreenRenderCanvas

        canvas = OffscreenRenderCanvas(size=(640, 480), pixel_ratio=2)
        renderer = pygfx.renderers.WgpuRenderer(canvas)

        canvas.request_draw(lambda: renderer.render(self._pygfx_scene, self._pygfx_cam))
        return np.asarray(canvas.draw())
