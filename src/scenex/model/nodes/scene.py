from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

from .node import Node

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from .node import NodeKwargs


class Scene(Node):
    """A Root node for a scene graph.

    This really isn't anything more than a regular Node, but it's an explicit
    marker that this node is the root of a scene graph.
    """

    node_type: Literal["scene"] = "scene"

    def __init__(
        self,
        *,
        children: Iterable["Node | dict[str, Any]"] = (),
        **data: "Unpack[NodeKwargs]",
    ) -> None:
        super().__init__(children=children, **data)
