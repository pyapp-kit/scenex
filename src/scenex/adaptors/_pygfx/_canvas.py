from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard, cast

from scenex.adaptors._base import CanvasAdaptor
from scenex.events._auto import app
from scenex.events.events import _handle_event

from ._adaptor_registry import get_adaptor

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

        canvas_cls = RenderCanvas
        # HACK: Qt
        if canvas_cls.__module__.startswith("rendercanvas.qt"):
            from qtpy.QtCore import QSize
            from rendercanvas.auto import loop
            from rendercanvas.qt import QRenderWidget

            class _QRenderWidget(QRenderWidget):
                def sizeHint(self) -> QSize:
                    return QSize(self.width(), self.height())

            loop._rc_init()
            canvas_cls = _QRenderWidget
        self._canvas = canvas
        self._wgpu_canvas = canvas_cls()

        # FIXME: This seems to not work on my laptop, without external monitors.
        # The physical canvas size is still 625, 625...
        self._wgpu_canvas.set_logical_size(canvas.width, canvas.height)
        self._wgpu_canvas.set_title(canvas.title)
        self._views: list[model.View] = []
        for view in canvas.views:
            self._snx_add_view(view)
        self._filter = app().install_event_filter(
            self._wgpu_canvas, canvas, lambda e: _handle_event(canvas, e)
        )

    def _snx_get_native(self) -> BaseRenderCanvas:
        return self._wgpu_canvas

    def _snx_set_visible(self, arg: bool) -> None:
        # show the qt canvas we patched earlier in __init__
        if supports_hide_show(self._wgpu_canvas):
            self._wgpu_canvas.show()
        self._wgpu_canvas.request_draw(self._draw)

    def _draw(self) -> None:
        for view in self._views:
            cast("View", get_adaptor(view))._draw()

    def _snx_add_view(self, view: model.View) -> None:
        # This logic should go in the canvas node, I think
        self._views.append(view)

        # FIXME: Allow customization
        x = 0.0
        dx = float(self._wgpu_canvas.get_logical_size()[0]) / len(self._views)

        for view in self._views:
            view.layout.x = x
            view.layout.y = 0
            view.layout.width = dx
            view.layout.height = self._wgpu_canvas.get_logical_size()[1]  # type: ignore
            x += dx

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
        return cast("np.ndarray", canvas.draw())
