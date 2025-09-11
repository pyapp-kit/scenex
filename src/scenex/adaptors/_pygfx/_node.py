from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from scenex import model
from scenex.adaptors._base import NodeAdaptor, TNode

from ._adaptor_registry import get_adaptor

if TYPE_CHECKING:
    import pygfx

    from scenex.model import Transform

logger = logging.getLogger("scenex.adaptors.pygfx")
TObj = TypeVar("TObj", bound="pygfx.WorldObject")
TMat = TypeVar("TMat", bound="pygfx.Material")
TGeo = TypeVar("TGeo", bound="pygfx.Geometry")

BLEND_MODES = {
    model.BlendMode.OPAQUE: "solid",
    model.BlendMode.ALPHA: "auto",
    model.BlendMode.ADDITIVE: "add",
}


class Node(NodeAdaptor[TNode, TObj], Generic[TNode, TObj, TMat, TGeo]):
    """Node adaptor for pygfx Backend."""

    _pygfx_node: TObj
    _material: TMat
    _geometry: TGeo
    _name: str

    def _snx_set_name(self, arg: str) -> None:
        # not sure pygfx has a name attribute...
        # TODO: for that matter... do we need a name attribute?
        # Could this be entirely managed on the model side/
        self._name = arg

    def _snx_add_child(self, arg: model.Node) -> None:
        child_adaptor = cast("Node", get_adaptor(arg))
        self._pygfx_node.add(child_adaptor._pygfx_node)

    def _snx_remove_child(self, arg: model.Node) -> None:
        child_adaptor = cast("Node", get_adaptor(arg))
        self._pygfx_node.remove(child_adaptor._pygfx_node)

    def _snx_set_visible(self, arg: bool) -> None:
        self._pygfx_node.visible = arg

    def _snx_set_opacity(self, arg: float) -> None:
        if material := getattr(self, "_material", None):
            material.opacity = arg

    def _snx_set_order(self, arg: int) -> None:
        self._pygfx_node.render_order = arg

    def _snx_set_interactive(self, arg: bool) -> None:
        pass
        # this one requires knowledge of the controller
        # warnings.warn("interactive not implemented in pygfx backend", stacklevel=2)

    def _snx_set_transform(self, arg: Transform) -> None:
        # pygfx uses a transposed matrix relative to the model
        self._pygfx_node.local.matrix = arg.root.T

    def _snx_set_blending(self, arg: model.BlendMode) -> None:
        material = self._pygfx_node.material
        if material is None:
            # FIXME this node is a scene
            return
        # pygfx only supports node blend modes for versions >=0.13
        if hasattr(material, "alpha_mode"):
            material.alpha_mode = BLEND_MODES[arg]  # pyright: ignore
        else:
            warnings.warn(
                "Node blending not supported by this version of pygfx - ignoring",
                RuntimeWarning,
                stacklevel=2,
            )

    def _snx_add_node(self, node: model.Node) -> None:
        # create if it doesn't exist
        adaptor = cast("Node", get_adaptor(node))
        self._pygfx_node.add(adaptor._pygfx_node)

    def _snx_force_update(self) -> None:
        pass

    def _snx_block_updates(self) -> None:
        pass

    def _snx_unblock_updates(self) -> None:
        pass
