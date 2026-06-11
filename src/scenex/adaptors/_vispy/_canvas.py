from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard, cast

import numpy as np
from vispy.color import Color as Color

from scenex.adaptors._base import CanvasAdaptor
from scenex.app import GuiFrontend, app, determine_app

from ._adaptor_registry import get_adaptor

if TYPE_CHECKING:
    import cmap
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
        from vispy.scene import SceneCanvas, VisualNode

        self._canvas = SceneCanvas(
            title=canvas.title, size=(canvas.width, canvas.height)
        )
        # Qt RenderCanvas calls show() in its __init__ method, so we need to hide it
        if supports_hide_show(self._canvas.native):
            self._canvas.native.hide()
        self._views: list[model.View] = []
        for view in canvas.views:
            self._snx_add_view(view)
        self._filter = app().install_event_filter(self._canvas.native, canvas.handle)

        self._visual_to_node: dict[VisualNode, model.Node | None] = {}
        self._last_canvas_pos: tuple[float, float] | None = None
        self._model = canvas

    def _snx_get_native(self) -> Any:
        return self._canvas.native

    def _snx_set_visible(self, arg: bool) -> None:
        app().show(self._snx_get_native(), arg)

    def _draw(self) -> None:
        self._canvas.update()

    def _snx_add_view(self, view: model.View) -> None:
        if view in self._views:
            return

        vis_view = cast("View", get_adaptor(view))
        # NOTE: canvas.central_widget.add_widget exists but
        # messes with the layout constantly. The docs specify that setting the parent
        # directly also works.
        vis_view._vispy_viewbox.parent = self._canvas.central_widget

        cast("View", get_adaptor(view))._on_size_changed()
        self._views.append(view)

    def _snx_set_width(self, arg: int) -> None:
        """When the canvas size changes we need to tell the vispy viewbox about it."""
        self._canvas.size = self._model.size
        self._update_view_rects()

    def _snx_set_height(self, arg: int) -> None:
        """When the canvas size changes we need to tell the vispy viewbox about it."""
        self._canvas.size = self._model.size
        self._update_view_rects()

    def _update_view_rects(self) -> None:
        for view in self._views:
            cast("View", get_adaptor(view))._on_size_changed()

    def _snx_set_background_color(self, arg: cmap.Color | None) -> None:
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
        bgcolor: cmap.Color | None = None,
        crop: np.ndarray | tuple[int, int, int, int] | None = None,
        alpha: bool = True,
    ) -> np.ndarray:
        """Render a screenshot."""
        backend = determine_app()
        if backend == GuiFrontend.JUPYTER:
            # The jupyter_rfb backend uses some tricks, breaking SceneCanvas.render
            # That backend's CanvasBackend.get_frame() allows an alternative approach
            native = self._canvas.native
            # The canvas refuses to render unless it has been correctly sized.
            # Correct sizing happens through handling a resize event passed through
            # IPython events
            native.handle_event(
                {
                    "event_type": "resize",
                    "width": self._model.width,
                    "height": self._model.height,
                    "pixel_ratio": 1,
                }
            )
            # Post-resize, get the frame!
            return native.get_frame()  # type: ignore
        else:
            # Convert background color to vispy
            vispy_bgcolor = None
            if bgcolor is not None:
                vispy_bgcolor = Color(bgcolor.rgba)
            # To render in VisPy, we need the canvas' GL context to be current.
            # Within wx, a current context enforces IsShown() at the C++ level.
            # So we have to show it.
            was_visible = self._model.visible
            if backend == GuiFrontend.WX and not was_visible:
                self._snx_set_visible(True)
            # Render!
            img = np.asarray(
                self._canvas.render(
                    region=region,
                    size=size,
                    bgcolor=vispy_bgcolor,
                    crop=crop,
                    alpha=alpha,
                )
            )
            # Within wx, a current context enforces IsShown() at the C++ level.
            # Now that we've rendered, let's hide it if it wasn't visible to start with.
            if backend == GuiFrontend.WX and not was_visible:
                self._snx_set_visible(False)
            return img
