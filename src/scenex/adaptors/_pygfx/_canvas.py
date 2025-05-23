from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard, cast

from scenex.adaptors._base import CanvasAdaptor

from ._adaptor_registry import adaptors

if TYPE_CHECKING:
    import numpy as np
    from cmap import Color
    from rendercanvas.base import BaseRenderCanvas

    from scenex import model

    from ._view import View

    class SupportsHideShow(BaseRenderCanvas):
        def show(self) -> None: ...
        def hide(self) -> None: ...


def supports_hide_show(obj: Any) -> TypeGuard[SupportsHideShow]:
    return hasattr(obj, "show") and hasattr(obj, "hide")


class Canvas(CanvasAdaptor):
    """Canvas interface for pygfx Backend."""

    def __init__(self, canvas: model.Canvas, **backend_kwargs: Any) -> None:
        from rendercanvas.auto import RenderCanvas

        self._wgpu_canvas = RenderCanvas()
        # Qt RenderCanvas calls show() in its __init__ method, so we need to hide it
        if supports_hide_show(self._wgpu_canvas):
            self._wgpu_canvas.hide()

        self._wgpu_canvas.set_logical_size(canvas.width, canvas.height)
        self._wgpu_canvas.set_title(canvas.title)
        self._views = canvas.views

    def _snx_get_native(self) -> BaseRenderCanvas:
        return self._wgpu_canvas

    def _snx_set_visible(self, arg: bool) -> None:
        # show the qt canvas we patched earlier in __init__
        if supports_hide_show(self._wgpu_canvas):
            self._wgpu_canvas.show()
        self._wgpu_canvas.request_draw(self._draw)

    def _draw(self) -> None:
        for view in self._views:
            adaptor = cast("View", adaptors.get_adaptor(view, create=True))
            adaptor._draw()

    def _snx_add_view(self, view: model.View) -> None:
        pass
        # adaptor = cast("View", view.backend_adaptor())
        # adaptor._pygfx_cam.set_viewport(self._viewport)
        # self._views.append(adaptor)

    def _snx_set_width(self, arg: int) -> None:
        _, height = cast("tuple[float, float]", self._wgpu_canvas.get_logical_size())
        self._wgpu_canvas.set_logical_size(arg, height)

    def _snx_set_height(self, arg: int) -> None:
        width, _ = cast("tuple[float, float]", self._wgpu_canvas.get_logical_size())
        self._wgpu_canvas.set_logical_size(width, arg)

    def _snx_set_background_color(self, arg: Color | None) -> None:
        # not sure if pygfx has both a canavs and view background color...
        pass

    def _snx_set_title(self, arg: str) -> None:
        self._wgpu_canvas.set_title(arg)

    def _snx_close(self) -> None:
        """Close canvas."""
        self._wgpu_canvas.close()

    def _snx_render(self) -> np.ndarray:
        """Render to offscreen buffer."""
        from rendercanvas.offscreen import OffscreenRenderCanvas

        # not sure about this...
        # w, h = self._wgpu_canvas.get_logical_size()
        canvas = OffscreenRenderCanvas(size=(640, 480), pixel_ratio=2)
        canvas.request_draw(self._draw)
        canvas.force_draw()
        breakpoint()
        return cast("np.ndarray", canvas.draw())
