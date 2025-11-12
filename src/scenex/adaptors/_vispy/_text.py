"""Vispy Text adaptor for SceneX Text node."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vispy.visuals
from vispy import scene

from scenex.adaptors._base import TextAdaptor

from ._node import Node

if TYPE_CHECKING:
    import cmap

    from scenex.model import BlendMode
    from scenex.model import Text as TextModel


class Text(Node, TextAdaptor):
    """vispy backend text adaptor."""

    _vispy_node: vispy.visuals.TextVisual

    def __init__(self, text: TextModel, **backend_kwargs: Any) -> None:
        self._model = text
        # create a vispy Text visual
        self._vispy_node = scene.Text(
            text=text.text, color=text.color.hex, font_size=text.size
        )

    def _snx_set_text(self, arg: str) -> None:
        self._vispy_node.text = arg

    def _snx_set_color(self, arg: cmap.Color) -> None:
        self._vispy_node.color = arg.hex

    def _snx_set_size(self, arg: int) -> None:
        self._vispy_node.font_size = arg

    def _snx_set_blending(self, arg: BlendMode) -> None:
        # Blending text makes it look very blocky
        pass
