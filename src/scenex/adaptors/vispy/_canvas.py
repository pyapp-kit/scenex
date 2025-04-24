from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard, cast

import numpy as np

from scenex.adaptors.base import CanvasAdaptor

from ._adaptor_registry import adaptors, get_adaptor

if TYPE_CHECKING:
    from cmap import Color
    from rendercanvas.auto import RenderCanvas
    from rendercanvas.base import BaseRenderCanvas

    from scenex import model

    from ._view import View

    class SupportsHideShow(BaseRenderCanvas):
        def show(self) -> None: ...
        def hide(self) -> None: ...


def supports_hide_show(obj: Any) -> TypeGuard[SupportsHideShow]:
    return hasattr(obj, "show") and hasattr(obj, "hide")


class Canvas(CanvasAdaptor):
    """Canvas interface for vispy Backend."""

    def __init__(self, canvas: model.Canvas, **backend_kwargs: Any) -> None:
        from vispy.scene import SceneCanvas, Grid
        from vispy.scene import Rectangle

        self._canvas = SceneCanvas(
            title=canvas.title,
            size=(canvas.width, canvas.height)
        )
        # rect = Rectangle(
        #     center=[100, 100],
        #     color="red",
        #     border_color="white",
        #     width=200,
        #     height=200,
        # )
        # rect.parent = self._canvas.scene
        # Qt RenderCanvas calls show() in its __init__ method, so we need to hide it
        if supports_hide_show(self._canvas.native):
            self._canvas.native.hide()
        self._grid = cast(Grid, self._canvas.central_widget.add_grid())
        for view in canvas.views:
            self._grid.add_widget(get_adaptor(view)._vispy_viewbox)
        self._views = canvas.views

    def _snx_get_native(self) -> RenderCanvas:
        return self._canvas.native

    def _snx_set_visible(self, arg: bool) -> None:
        # show the qt canvas we patched earlier in __init__
        if supports_hide_show(self._canvas.native):
            self._canvas.show()
        # TODO: Is this needed?
        # self._wgpu_canvas.request_draw(self._draw)

    def _draw(self) -> None:
        self._canvas.update()
        # for view in self._views:
        #     adaptor = cast("View", adaptors.get_adaptor(view))
        #     adaptor._draw()

    def _snx_add_view(self, view: model.View) -> None:
        self._grid.add_widget(get_adaptor(view)._vispy_viewbox)
        # adaptor = cast("View", view.backend_adaptor())
        # adaptor._pygfx_cam.set_viewport(self._viewport)
        # self._views.append(adaptor)

    def _snx_set_width(self, arg: int) -> None:
        self._canvas.size = (self._canvas.size[0], arg)

    def _snx_set_height(self, arg: int) -> None:
        self._canvas.size = (arg, self._canvas.size[1])

    def _snx_set_background_color(self, arg: Color | None) -> None:
        if arg is None:
            self._canvas.bgcolor = "black"
        else:
            self._canvas.bgcolor = arg.rgba

    def _snx_set_title(self, arg: str) -> None:
        self._canvas.title = arg

    def _snx_close(self) -> None:
        """Close canvas."""
        self._canvas.close()

    def _snx_render(
        self,
        region: tuple[int, int, int, int] | None = None,
        size: tuple[int, int] | None = None,
        bgcolor: Color | None = None,
        crop: np.ndarray | tuple[int, int, int, int] | None = None,
        alpha: bool = True,
    ) -> np.ndarray:
        """Render to screenshot."""
        # VERY sure about this...
        return np.asarray(self._canvas.render())
