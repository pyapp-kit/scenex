try:
    from imgui_bundle import imgui
    from wgpu.utils.imgui import ImguiRenderer
except ImportError as e:
    raise ImportError(
        "imgui_bundle and pygfx are required for imgui controls. "
        "Please install scenex with 'pip install scenex[imgui]'."
    ) from e

import logging
import types
from typing import Any, get_args, get_origin

import annotated_types
import cmap
from imgui_bundle import imgui
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from rendercanvas import BaseRenderCanvas

import scenex as snx
from scenex.adaptors._pygfx._canvas import Canvas as PygfxCanvasAdaptor
from scenex.model._view import View

logger = logging.getLogger("scenex.imgui")


def add_imgui_controls(view: View) -> None:
    """Add imgui controls to the given canvas."""
    snx_canvas_model = view.canvas
    snx_canvas_adaptor = snx_canvas_model._get_adaptor()
    render_canv = snx_canvas_model._get_native()

    if not (
        isinstance(snx_canvas_adaptor, PygfxCanvasAdaptor)
        and isinstance(snx_canvas_model._get_native(), BaseRenderCanvas)
    ):
        raise NotImplementedError(
            "Imgui controls can currently only be added to a canvas backed by pygfx."
        )

    def _get_imgui_renderer(canvas: "BaseRenderCanvas") -> ImguiRenderer:
        """Get an ImguiRenderer for the given canvas."""
        # lots of hacks in here...
        render_canv.set_update_mode("continuous")
        ctx = render_canv.get_context("wgpu")
        return ImguiRenderer(device=ctx._config["device"], canvas=render_canv)

    imgui_renderer = _get_imgui_renderer(render_canv)

    @imgui_renderer.set_gui  # type: ignore
    def _update_gui() -> imgui.ImDrawData:
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0), imgui.Cond_.appearing)  # type: ignore
        imgui.set_next_window_size((300, 0), imgui.Cond_.appearing)  # type: ignore
        render_imgui_view_controls(view)
        imgui.end_frame()
        imgui.render()
        return imgui.get_draw_data()

    @render_canv.request_draw  # type: ignore
    def _update() -> None:
        snx_canvas_adaptor._draw()
        imgui_renderer.render()


def _min_max(meta: list[Any], eps: float = 0) -> tuple[float | None, float | None]:
    mi, ma = None, None
    for item in meta:
        if isinstance(item, annotated_types.Ge):
            mi = float(item.ge)  # type: ignore
        elif isinstance(item, annotated_types.Le):
            ma = float(item.le)  # type: ignore
        elif isinstance(item, annotated_types.Gt):
            mi = float(item.gt) + eps  # type: ignore
        elif isinstance(item, annotated_types.Lt):
            ma = float(item.lt) - eps  # type: ignore
        elif isinstance(item, annotated_types.Interval):
            mi, ma = _min_max(list(item))
    return mi, ma


def render_field_widget(name: str, finfo: FieldInfo, value: Any) -> tuple[bool, Any]:
    """Render an imgui widget for field, based on its type."""
    if finfo.repr is False:
        return (False, value)
    annotation = finfo.annotation
    if get_origin(annotation) is types.UnionType:
        args = get_args(annotation)
        if len(args) == 2 and types.NoneType in args:  # Optional[T]
            annotation = next(arg for arg in args if arg is not types.NoneType)

    if isinstance(annotation, type):
        if annotation is bool:
            return imgui.checkbox(name, value)
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
    return (False, value)


def render_imgui_model_controls(model: BaseModel) -> None:
    """Update the GUI with the current state."""
    fields = type(model).model_fields
    for field, value in model:
        try:
            changed, val = render_field_widget(field, fields[field], value)
        except Exception as e:
            logger.debug("Error creating imgui widget for %s: %s", field, e)
        else:
            if changed:
                try:
                    setattr(model, field, val)
                except ValidationError as e:
                    logger.debug("Validation error in imgui widget setter: %s", e)


def render_imgui_view_controls(view: snx.View) -> None:
    """Update the GUI with the current state."""
    imgui.begin("Controls", None, imgui.WindowFlags_.always_auto_resize)  # type: ignore
    for i, child in enumerate(view.scene.children):
        imgui.push_id(i)
        name = repr(child.name) if child.name else f"@ {child._model_id.hex[:5]}"
        if imgui.collapsing_header(f"{type(child).__name__} {name}"):
            render_imgui_model_controls(child)
        imgui.pop_id()
    imgui.end()
