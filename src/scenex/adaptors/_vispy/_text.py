"""Vispy Text adaptor for SceneX Text node."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vispy.visuals
from vispy import scene

from scenex.adaptors._base import TextAdaptor

from ._node import Node

if TYPE_CHECKING:
    import cmap
    from vispy.util.event import Event

    from scenex.model import BlendMode
    from scenex.model import Text as TextModel


class Text(Node, TextAdaptor):
    """vispy backend text adaptor."""

    _vispy_node: vispy.visuals.TextVisual

    def __init__(self, text: TextModel, **backend_kwargs: Any) -> None:
        self._model = text
        # create a vispy Text visual
        self._vispy_node = scene.Text(text=text.text, color=text.color.hex)
        # HACK: We need to tap into the current DPI of the canvas to convert pixels to
        # point size. I don't know of a better way to do this.
        self._vispy_node.transforms.changed.connect(self._update_dpi)  # pyright: ignore[reportOptionalMemberAccess]
        # Set font size separately to ensure conversion
        self._snx_set_size(text.size)

    def _snx_set_text(self, arg: str) -> None:
        self._vispy_node.text = arg

    def _snx_set_color(self, arg: cmap.Color) -> None:
        self._vispy_node.color = arg.hex

    def _snx_set_size(self, arg: int) -> None:
        # Vispy works in points and we work in pixels, so we need to convert
        dpi = self._vispy_node.transforms.dpi or 96  # pyright: ignore[reportOptionalMemberAccess]
        # This formula is an inversion of what vispy does in TextVisual._prepare_draw
        font_size = (arg * 72) / dpi
        # Only update if necessary to avoid triggering a redraw
        if self._vispy_node.font_size != font_size:
            self._vispy_node.font_size = font_size

    def _snx_set_antialias(self, arg: bool) -> None:
        # vispy's TextVisual uses SDF-based rendering and cannot be tuned; antialiasing
        # is always applied internally.
        # TODO: Consider logging something here? There are bigger questions around how
        # scenex should handle something that a backend doesn't support. For now let's
        # be quiet.
        pass

    def _snx_set_blending(self, arg: BlendMode) -> None:
        # Blending text makes it look very blocky
        pass

    def _update_dpi(self, event: Event) -> None:
        self._snx_set_size(self._model.size)
