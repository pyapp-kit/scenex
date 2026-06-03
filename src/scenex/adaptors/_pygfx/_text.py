"""Vispy Text adaptor for SceneX Text node."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pygfx

from scenex.adaptors._base import TextAdaptor

from ._node import Node

if TYPE_CHECKING:
    import cmap

    from scenex.model import Text as TextModel


class Text(Node, TextAdaptor):
    """pygfx backend text adaptor."""

    _material: pygfx.TextMaterial
    _pygfx_node: pygfx.Text

    def __init__(self, text: TextModel, **backend_kwargs: Any) -> None:
        self._model = text
        self._material = pygfx.TextMaterial(
            color=text.color.hex,
            # This value has model render order win for coplanar objects
            depth_compare="<=",
            aa=text.antialias,
        )
        # create a pygfx Text visual
        self._pygfx_node = pygfx.Text(
            text=text.text,
            material=self._material,
            font_size=text.size,
            screen_space=True,
        )

    def _snx_set_text(self, arg: str) -> None:
        self._pygfx_node.set_text(arg)

    def _snx_set_color(self, arg: cmap.Color) -> None:
        self._material.color = arg.hex

    def _snx_set_size(self, arg: int) -> None:
        self._pygfx_node.font_size = arg

    def _snx_set_antialias(self, arg: bool) -> None:
        self._material.aa = arg
