from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import numpy as np
import vispy.scene

from scenex import model
from scenex.adaptors._base import NodeAdaptor, TNode

from ._adaptor_registry import get_adaptor

if TYPE_CHECKING:
    from scenex.model import Transform


TObj = TypeVar("TObj", bound="vispy.scene.Node")

BLEND_MODES = {
    model.BlendMode.OPAQUE: "opaque",
    model.BlendMode.ALPHA: "translucent",
    model.BlendMode.ADDITIVE: "additive",
}


class Node(NodeAdaptor[TNode, TObj], Generic[TNode, TObj]):
    """Node adaptor for pygfx Backend."""

    _vispy_node: TObj

    def _snx_set_name(self, arg: str) -> None:
        self._vispy_node.name = arg

    def _snx_add_child(self, child: model.Node) -> None:
        # create if it doesn't exist
        child_adaptor = cast(
            "Node[Any, vispy.scene.Node]", get_adaptor(child, create=True)
        )
        child_adaptor._vispy_node.parent = self._vispy_node

    def _snx_remove_child(self, child: model.Node) -> None:
        # create if it doesn't exist
        child_adaptor = cast(
            "Node[Any, vispy.scene.Node]", get_adaptor(child, create=True)
        )
        child_adaptor._vispy_node.parent = None

    def _snx_set_visible(self, arg: bool) -> None:
        self._vispy_node.visible = arg

    def _snx_set_opacity(self, arg: float) -> None:
        self._vispy_node.opacity = arg

    def _snx_set_order(self, arg: int) -> None:
        self._vispy_node.order = arg

    def _snx_set_interactive(self, arg: bool) -> None:
        pass

    def _snx_set_transform(self, arg: Transform) -> None:
        self._vispy_node.transform = vispy.scene.transforms.MatrixTransform(
            np.asarray(arg)
        )

    def _snx_set_blending(self, arg: model.BlendMode) -> None:
        if hasattr(self._vispy_node, "set_gl_state"):
            if arg == model.BlendMode.OPAQUE:
                # for opaque, we need to disable blending
                self._vispy_node.set_gl_state(None, blend=False)  # pyright: ignore
            else:
                self._vispy_node.set_gl_state(BLEND_MODES[arg])  # pyright: ignore

    def _snx_add_node(self, node: model.Node) -> None:
        # create if it doesn't exist
        adaptor = cast("Node", get_adaptor(node))
        adaptor._vispy_node.parent = self._vispy_node

    def _snx_force_update(self) -> None:
        pass

    def _snx_block_updates(self) -> None:
        pass

    def _snx_unblock_updates(self) -> None:
        pass
