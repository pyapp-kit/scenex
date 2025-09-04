from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard, cast

import numpy as np

from scenex.adaptors._base import CanvasAdaptor
from scenex.app import app

from ._adaptor_registry import get_adaptor

if TYPE_CHECKING:
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
    """Canvas interface for vispy Backend."""

    def __init__(self, canvas: model.Canvas, **backend_kwargs: Any) -> None:
        from vispy.scene import Grid, SceneCanvas, VisualNode

        self._canvas = SceneCanvas(
            title=canvas.title, size=(canvas.width, canvas.height)
        )
        # Qt RenderCanvas calls show() in its __init__ method, so we need to hide it
        if supports_hide_show(self._canvas.native):
            self._canvas.native.hide()
        self._grid = cast("Grid", self._canvas.central_widget.add_grid())
        self._views: list[model.View] = []
        for view in canvas.views:
            self._snx_add_view(view)
        self._filter = app().install_event_filter(self._canvas.native, canvas)

        self._visual_to_node: dict[VisualNode, model.Node | None] = {}
        self._last_canvas_pos: tuple[float, float] | None = None

    def _snx_get_native(self) -> Any:
        return self._canvas.native

    def _snx_set_visible(self, arg: bool) -> None:
        # show the qt canvas we patched earlier in __init__
        app().show(self, arg)
        # HACK
        self._canvas.set_current()
        app().process_events()

    def _draw(self) -> None:
        self._canvas.update()

    def _snx_add_view(self, view: model.View) -> None:
        if view in self._views:
            return
        self._views.append(view)
        # FIXME: Allow customization
        x = 0.0
        dx = float(self._canvas.size[0]) / len(self._views)

        for view in self._views:
            view.layout.x = x
            view.layout.y = 0
            view.layout.width = dx
            view.layout.height = self._canvas.size[1]
            x += dx

        self._grid.add_widget(cast("View", get_adaptor(view))._vispy_viewbox)
        get_adaptor(view.camera)._set_view(view.layout.width, view.layout.height)  # type:ignore
        # adaptor = get_adaptor(view)
        # self._grid.add_widget(adaptor._snx_get_native())
        # # HACK: Update view size by passing the existing camera
        # self._grid._prepare_draw(adaptor._snx_get_native())
        # cam_adaptor = get_adaptor(view.camera)
        # cam_adaptor._set_view(adaptor._vispy_viewbox)  # type: ignore

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
        vispy_bgcolor = None
        if bgcolor and bgcolor.name:
            from vispy.color import Color as _Color

            vispy_bgcolor = _Color(bgcolor.name)
        return np.asarray(
            self._canvas.render(
                region=region, size=size, bgcolor=vispy_bgcolor, crop=crop, alpha=alpha
            )
        )
