from __future__ import annotations

import warnings

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
from scenex.app.events import (
    Event,
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
    WheelEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic.fields import FieldInfo

    import scenex as snx
    from scenex.adaptors._base import CanvasAdaptor
    from scenex.model._view import View

logger = logging.getLogger("scenex.imgui")

_REGISTERED_COLORMAPS: set[str] = set()


def add_imgui_controls(view: View) -> None:
    """Add an interactive ImGui control panel to a view.

    Creates an overlay control panel that allows real-time manipulation of view
    properties and scene node attributes through automatically generated widgets.
    The panel displays collapsible sections for the view and each child node in
    the scene, with widgets dynamically created based on Pydantic field types.

    Parameters
    ----------
    view : View
        The view to control.

    Raises
    ------
    NotImplementedError
        If the view is not using the pygfx backend.
    RuntimeError
        If the pygfx renderer has not been initialized yet.
    ImportError
        If required dependencies (imgui_bundle, pygfx) are not installed.

    Notes
    -----
    - Only works with the pygfx backend
    - The control panel is rendered as an overlay on the canvas. It is not (currently)
      restricted to a specific area of the canvas
    - Current architecture necessitates this function be called AFTER setting up camera
      controllers and/or view event filters. All view events are intercepted and may not
      propagate to the user's view filter or camera filter, but a best attempt is made
      to propagate events that do not interact with the ImGui control panel.
    - Widgets are automatically generated from Pydantic field metadata:
        * Literal types → dropdown menus
        * bool → checkbox
        * int/float with bounds → slider
        * int/float without bounds → input field
        * Color → color picker
        * Colormap → colormap preview button

    Examples
    --------
    Basic usage with an image::

        import scenex as snx
        from scenex.imgui import add_imgui_controls

        image = snx.Image(data=my_array)
        view = snx.View(scene=snx.Scene(children=[image]))
        add_imgui_controls(view)
        snx.show(view)
        snx.run()

    With multiple nodes::

        scene = snx.Scene(
            children=[
                snx.Image(data=data1),
                snx.Points(coords=points),
                snx.Mesh(vertices=verts, faces=faces),
            ]
        )
        view = snx.View(scene=scene)
        add_imgui_controls(view)
        snx.show(view)
        snx.run()

    The control panel will show sections for:
    - View properties (camera, layout, etc.)
    - Image node (colormap, clims, opacity, etc.)
    - Points node (size, color, symbol, etc.)
    - Mesh node (color, opacity, blending, etc.)
    """
    snx_canvas_model = view.canvas
    try:
        snx_canvas_adaptor = snx_canvas_model._get_adaptors(backend="pygfx")[0]
        snx_view_adaptor = view._get_adaptors(backend="pygfx")[0]
    except (KeyError, IndexError):
        warnings.warn(
            "No pygfx adaptor found view/canvas; cannot add imgui controls.",
            stacklevel=2,
        )
        return

    render_canv = cast("CanvasAdaptor", snx_canvas_adaptor)._snx_get_native()

    if not (
        isinstance(snx_canvas_adaptor, PygfxCanvasAdaptor)
        and isinstance(snx_view_adaptor, PygfxViewAdaptor)
        and isinstance(render_canv, BaseRenderCanvas)
    ):
        raise NotImplementedError(
            "Imgui controls can currently only be added to a canvas backed by pygfx."
        )
    if not snx_canvas_adaptor._renderer:
        raise RuntimeError("The pygfx renderer has not been initialized yet.")

    imgui_renderer = ImguiRenderer(
        device=snx_canvas_adaptor._renderer.device,
        canvas=render_canv,  # pyright: ignore[reportArgumentType] (incorrect hint)
    )

    if implot.get_current_context() is None:
        implot.create_context()  # must run after ImGui context exists

    @imgui_renderer.set_gui  # type: ignore [untyped-decorator]
    def _update_gui() -> None:
        render_imgui_view_controls(view)

    @render_canv.request_draw
    def _update() -> None:
        snx_canvas_adaptor._draw()
        imgui_renderer.render()

    class ImguiEventFilter:
        internal_filter: Callable[[Event], bool] | None = None

        def __call__(self, event: Event) -> bool:
            # NOTE: As the scenex event system matures
            # It may capture more events (notably, keypresses).
            # We will have to intercept scenex events here if that occurs
            if isinstance(event, MouseMoveEvent):
                move_dict = {"x": event.canvas_pos[0], "y": event.canvas_pos[1]}
                imgui_renderer._on_mouse_move(move_dict)
                if move_dict.get("stop_propagation", False):
                    return True
            if isinstance(event, MousePressEvent):
                btn = imgui_filter.convert_btn(event.buttons)
                press_dict = {"button": btn, "event_type": "pointer_down"}
                imgui_renderer._on_mouse(press_dict)
                if press_dict.get("stop_propagation", False):
                    return True
            if isinstance(event, MouseReleaseEvent):
                btn = imgui_filter.convert_btn(event.buttons)
                release_dict = {"button": btn, "event_type": "pointer_up"}
                imgui_renderer._on_mouse(release_dict)
                if release_dict.get("stop_propagation", False):
                    return True
            if isinstance(event, WheelEvent):
                # FIXME: Validate correct delta sign
                wheel_dict = {"dx": event.angle_delta[0], "dy": event.angle_delta[1]}
                imgui_renderer._on_wheel(wheel_dict)
                if wheel_dict.get("stop_propagation", False):
                    return True

            if self.internal_filter is None:
                return False
            return self.internal_filter(event)

        def convert_btn(self, btn: MouseButton) -> int:
            if btn & MouseButton.LEFT:
                return 1
            if btn & MouseButton.RIGHT:
                return 2
            if btn & MouseButton.MIDDLE:
                return 3
            return 0

    imgui_filter = ImguiEventFilter()
    imgui_filter.internal_filter = view.set_event_filter(imgui_filter)


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
    imgui.set_next_window_pos((0, 0), imgui.Cond_.appearing)
    imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
    imgui.begin("Controls", None, imgui.WindowFlags_.always_auto_resize)
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
