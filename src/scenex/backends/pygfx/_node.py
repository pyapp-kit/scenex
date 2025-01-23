from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from scenex.model.nodes import node as core_node

from ._adaptor_registry import get_adaptor

if TYPE_CHECKING:
    from pygfx.geometries import Geometry
    from pygfx.materials import Material
    from pygfx.objects import WorldObject

    from scenex.model import Transform


class Node(core_node.NodeAdaptor):
    """Node adaptor for pygfx Backend."""

    _pygfx_node: WorldObject
    _material: Material
    _geometry: Geometry
    _name: str

    def _snx_get_native(self) -> Any:
        return self._pygfx_node

    def _snx_set_name(self, arg: str) -> None:
        # not sure pygfx has a name attribute...
        # TODO: for that matter... do we need a name attribute?
        # Could this be entirely managed on the model side/
        self._name = arg

    def _snx_set_parent(self, parent: core_node.Node | None) -> None:
        if parent is None:
            self._pygfx_node._reset_parent()
        else:
            parent_adaptor = cast("Node", get_adaptor(parent))
            parent_adaptor._pygfx_node.add(self._pygfx_node)

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

    def _vis_add_node(self, node: core_node.Node) -> None:
        # create if it doesn't exist
        adaptor = cast("Node", get_adaptor(node))
        self._pygfx_node.add(adaptor._snx_get_native())

    def _vis_force_update(self) -> None:
        pass

    def _vis_block_updates(self) -> None:
        pass

    def _vis_unblock_updates(self) -> None:
        pass
