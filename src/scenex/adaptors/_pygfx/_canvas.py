from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard, cast

from scenex.adaptors._base import CanvasAdaptor
from scenex.app import GuiFrontend, app, determine_app

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


def _rendercanvas_class() -> BaseRenderCanvas:
    """Obtains the appropriate class for the current GUI backend.

    Explicit since PyGFX's backend selection process may be different from ours.
    """
    frontend = determine_app()

    if frontend == GuiFrontend.QT:
        from qtpy.QtCore import QSize  # pyright: ignore[reportMissingImports]
        from rendercanvas.qt import QRenderWidget

        class _QRenderWidget(QRenderWidget):
            def sizeHint(self) -> QSize:
                return QSize(self.width(), self.height())

        # Init Qt Application - otherwise we can't create the widget
        app()
        return _QRenderWidget()  # type: ignore[no-untyped-call]

    if frontend == GuiFrontend.JUPYTER:
        import rendercanvas.jupyter

        return rendercanvas.jupyter.JupyterRenderCanvas()
    if frontend == GuiFrontend.WX:
        import rendercanvas.wx

        return rendercanvas.wx.WxRenderCanvas()

    raise ValueError("No suitable render canvas found")


class Canvas(CanvasAdaptor):
    """Canvas interface for pygfx Backend."""

    def __init__(self, canvas: model.Canvas, **backend_kwargs: Any) -> None:
        self._canvas = canvas
        self._wgpu_canvas = _rendercanvas_class()

        # FIXME: This seems to not work on my laptop, without external monitors.
        # The physical canvas size is still 625, 625...
        self._wgpu_canvas.set_logical_size(canvas.width, canvas.height)
        self._wgpu_canvas.set_title(canvas.title)
        self._views: list[model.View] = []
        for view in canvas.views:
            self._snx_add_view(view)
        self._filter = app().install_event_filter(self._snx_get_native(), canvas)

    def _snx_get_native(self) -> Any:
        if subwdg := getattr(self._wgpu_canvas, "_subwidget", None):
            # wx backend has a _subwidget attribute that is the actual widget
            return subwdg
        return self._wgpu_canvas

    def _snx_set_visible(self, arg: bool) -> None:
        app().show(self, arg)
        self._wgpu_canvas.request_draw(self._draw)

    def _draw(self) -> None:
        for view in self._views:
            cast("View", get_adaptor(view))._draw()

    def _snx_add_view(self, view: model.View) -> None:
        # This logic should go in the canvas node, I think
        if view in self._views:
            return
        self._views.append(view)

        view_adaptor = cast("View", get_adaptor(view))
        view_adaptor._set_pygfx_canvas(
            self._wgpu_canvas, self._canvas.width, self._canvas.height
        )

    def _snx_set_width(self, arg: int) -> None:
        width, height = cast(
            "tuple[float, float]", self._wgpu_canvas.get_logical_size()
        )
        # FIXME: For some reason, on wx the size has already been updated, and
        # updating it again causes erratic resizing behavior
        if width != arg:
            with app().block_events(self._snx_get_native()):
                self._wgpu_canvas.set_logical_size(arg, height)

    def _snx_set_height(self, arg: int) -> None:
        width, height = cast(
            "tuple[float, float]", self._wgpu_canvas.get_logical_size()
        )
        # FIXME: For some reason, on wx the size has already been updated, and
        # updating it again causes erratic resizing behavior
        if height != arg:
            with app().block_events(self._snx_get_native()):
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
