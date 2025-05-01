from typing import TYPE_CHECKING, Any, Literal

from .node import Node

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Unpack

    from .node import NodeKwargs


class Scene(Node):
    """A root node for a scene graph.

    This really isn't anything more than a regular Node, but it's an explicit
    marker that this node is the root of a scene graph.
    """

    node_type: Literal["scene"] = "scene"

    # tell mypy and pyright that this takes children, just like Node
    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            children: Iterable["Node | dict[str, Any]"] = (),
            **data: "Unpack[NodeKwargs]",
        ) -> None: ...
