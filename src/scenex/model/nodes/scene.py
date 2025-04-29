from collections.abc import Sequence
from typing import Any, Literal

from .node import AnyNode, Node


class Scene(Node):
    """A Root node for a scene graph.

    This really isn't anything more than a regular Node, but it's an explicit
    marker that this node is the root of a scene graph.
    """

    node_type: Literal["scene"] = "scene"

    def __init__(self, children: Sequence["AnyNode"] = (), **data: Any) -> None:
        super().__init__(children=children, **data)
