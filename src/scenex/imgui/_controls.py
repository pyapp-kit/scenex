from __future__ import annotations

import numpy as np

try:
    from imgui_bundle import imgui, implot
    from wgpu.utils.imgui import ImguiRenderer
except ImportError as e:
    raise ImportError(
        "imgui_bundle and pygfx are required for imgui controls. "
        "Please install scenex with 'pip install scenex[imgui]'."
    ) from e
import logging
import types
from typing import TYPE_CHECKING, Any, Literal, cast, get_args, get_origin

import annotated_types
import cmap
from imgui_bundle import imgui
from pydantic import BaseModel, ValidationError
from rendercanvas import BaseRenderCanvas

from scenex.adaptors._pygfx._canvas import Canvas as PygfxCanvasAdaptor
from scenex.adaptors._pygfx._view import View as PygfxViewAdaptor

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    import scenex as snx
    from scenex.adaptors._base import CanvasAdaptor
    from scenex.model._view import View

logger = logging.getLogger("scenex.imgui")

_REGISTERED_COLORMAPS: set[str] = set()


def add_imgui_controls(view: View) -> None:
    """Add imgui controls to the given canvas."""
    snx_canvas_model = view.canvas
    snx_canvas_adaptor = snx_canvas_model._get_adaptors(backend="pygfx")[0]
    snx_view_adaptor = view._get_adaptors(backend="pygfx")[0]
    render_canv = cast("CanvasAdaptor", snx_canvas_adaptor)._snx_get_native()

    if not (
        isinstance(snx_canvas_adaptor, PygfxCanvasAdaptor)
        and isinstance(snx_view_adaptor, PygfxViewAdaptor)
        and isinstance(render_canv, BaseRenderCanvas)
    ):
        raise NotImplementedError(
            "Imgui controls can currently only be added to a canvas backed by pygfx."
        )
    if not snx_view_adaptor._renderer:
        raise RuntimeError("The pygfx renderer has not been initialized yet.")

    imgui_renderer = ImguiRenderer(
        device=snx_view_adaptor._renderer.device,
        canvas=render_canv,  # pyright: ignore[reportArgumentType] (incorrect hint)
    )

    if implot.get_current_context() is None:
        implot.create_context()  # must run after ImGui context exists

    @imgui_renderer.set_gui  # type: ignore [misc]
    def _update_gui() -> imgui.ImDrawData:
        imgui.new_frame()
        render_imgui_view_controls(view)
        imgui.end_frame()
        imgui.render()
        return imgui.get_draw_data()

    @render_canv.request_draw
    def _update() -> None:
        snx_canvas_adaptor._draw()
        imgui_renderer.render()


def _min_max(meta: list[Any], eps: float = 0) -> tuple[float | None, float | None]:
    mi, ma = None, None
    for item in meta:
        if isinstance(item, annotated_types.Ge):
            mi = float(item.ge)  # type: ignore[arg-type]
        elif isinstance(item, annotated_types.Le):
            ma = float(item.le)  # type: ignore[arg-type]
        elif isinstance(item, annotated_types.Gt):
            mi = float(item.gt) + eps  # type: ignore[arg-type]
        elif isinstance(item, annotated_types.Lt):
            ma = float(item.lt) - eps  # type: ignore[arg-type]
        elif isinstance(item, annotated_types.Interval):
            mi, ma = _min_max(list(item))
    return mi, ma


def render_field_widget(name: str, finfo: FieldInfo, value: Any) -> tuple[bool, Any]:
    """Render an imgui widget for field, based on its type."""
    if finfo.repr is False:
        return (False, value)
    annotation = finfo.annotation
    origin = get_origin(annotation)
    optional = False
    if origin is Literal:
        args = get_args(annotation)
        if len(args) == 1:  # constant value
            return (False, value)
        # render a combo box
        items = [str(arg) for arg in args]
        idx = items.index(value) if value in items else 0
        changed, new_idx = imgui.combo(name, idx, items)
        return (changed, items[new_idx])
    elif origin is types.UnionType:
        args = get_args(annotation)
        if len(args) == 2 and types.NoneType in args:  # Optional[T]
            annotation = next(arg for arg in args if arg is not types.NoneType)
            optional = True

    if isinstance(annotation, type):
        if annotation is bool:
            return imgui.checkbox(name, value)
        if annotation is str:
            change, value = imgui.input_text(name, str(value or ""))
            if optional and value == "":
                value = None
            return change, value
        if issubclass(annotation, int):
            mi, ma = _min_max(finfo.metadata, eps=1)
            if mi is not None and ma is not None:
                return imgui.slider_int(name, value, int(mi), int(ma))
            return imgui.input_int(name, value)
        if issubclass(annotation, float):
            mi, ma = _min_max(finfo.metadata, eps=1e-5)
            if mi is not None and ma is not None:
                return imgui.slider_float(name, value, float(mi), float(ma))
            return imgui.input_float(name, value)
        if issubclass(annotation, cmap.Color):
            if not isinstance(value, cmap.Color):
                value = cmap.Color("transparent")
            changed, value = imgui.color_edit4(name, list(value.rgba))
            return changed, tuple(value)
        if isinstance(value, cmap.Colormap):
            w = imgui.calc_item_width()
            name = value.name.split(":")[-1]
            if name not in _REGISTERED_COLORMAPS:
                implot.add_colormap(name, value.lut().astype(np.float32))
                _REGISTERED_COLORMAPS.add(name)
            implot.push_colormap(name)
            if implot.colormap_button(name, (w, 20)):
                ...
            implot.pop_colormap()
            return (False, value)

    return (False, value)


def render_imgui_model_controls(model: BaseModel) -> None:
    """Update the GUI with the current state."""
    fields = type(model).model_fields
    for field, value in model:
        try:
            changed, val = render_field_widget(field, fields[field], value)
        except Exception as e:
            print(e)
            logger.debug("Error creating imgui widget for %s: %s", field, e)
        else:
            if changed:
                try:
                    setattr(model, field, val)
                except ValidationError as e:
                    logger.debug("Validation error in imgui widget setter: %s", e)


def render_imgui_view_controls(view: snx.View) -> None:
    """Update the GUI with the current state."""
    imgui.set_next_window_pos((0, 0), imgui.Cond_.appearing)  # type: ignore[arg-type]
    imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)  # type: ignore
    imgui.begin("Controls", None, imgui.WindowFlags_.always_auto_resize)  # type: ignore
    if imgui.collapsing_header("View"):
        render_imgui_model_controls(view)
    for i, child in enumerate(view.scene.children):
        imgui.push_id(i)
        header = f"{type(child).__name__} @ {child._model_id.hex[:5]}"
        if imgui.collapsing_header(header):
            render_imgui_model_controls(child)
        imgui.pop_id()
    imgui.end()
    imgui.pop_style_var()
