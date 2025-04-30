import logging
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Annotated, Any, Union, cast

from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    computed_field,
    model_serializer,
)

from scenex.model._base import EventedBase
from scenex.model.transform import Transform

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import TypedDict, Unpack

    from .camera import Camera
    from .image import Image
    from .points import Points
    from .scene import Scene

    class NodeKwargs(TypedDict, total=False):
        """TypedDict for Node kwargs."""

        name: str | None
        visible: bool
        interactive: bool
        opacity: float
        order: int
        transform: Transform | npt.ArrayLike


logger = logging.getLogger(__name__)


# improve me... Read up on: https://docs.pydantic.dev/latest/concepts/unions/
AnyNode = Annotated[
    Union["Image", "Points", "Camera", "Scene"], Field(discriminator="node_type")
]


class Node(EventedBase):
    """Base class for all nodes.  Also a `Container[Node]`.

    Do not instantiate this class directly. Use a subclass.  GenericNode may
    be used in place of Node.
    """

    # see computed fields below
    _parent: "AnyNode | None" = PrivateAttr(default=None)
    _children: list["AnyNode"] = PrivateAttr(default_factory=list)

    name: str | None = Field(default=None, description="Name of the node.")
    visible: bool = Field(default=True, description="Whether this node is visible.")
    interactive: bool = Field(
        default=False, description="Whether this node accepts mouse and touch events"
    )
    opacity: float = Field(default=1.0, ge=0, le=1, description="Opacity of this node.")
    order: int = Field(
        default=0,
        ge=0,
        description="A value used to determine the order in which nodes are drawn. "
        "Greater values are drawn later. Children are always drawn after their parent",
    )
    transform: Transform = Field(
        default_factory=Transform,
        description="Transform that maps the local coordinate frame to the coordinate "
        "frame of the parent.",
    )

    model_config = ConfigDict(extra="forbid")

    def __init__(
        self,
        *,
        children: Iterable["Node | dict[str, Any]"] = (),
        **data: "Unpack[NodeKwargs]",
    ) -> None:
        # prevent direct instantiation.
        # makes it easier to use NodeUnion without having to deal with self-reference.
        if type(self) is Node:
            raise TypeError("Node cannot be instantiated directly. Use a subclass.")

        super().__init__(**data)  # pyright: ignore[reportCallIssue]

        for ch in children:
            if not isinstance(ch, Node):
                ch = Node.model_validate(ch)
            self.add_child(ch)  # type: ignore [arg-type]

    @property
    def children(self) -> tuple["Node", ...]:
        """Return a tuple of the children of this node."""
        return tuple(self._children)

    def add_child(self, child: "AnyNode") -> None:
        """Add a child node to this node."""
        self._children.append(child)
        child._parent = cast("AnyNode", self)

    @computed_field  # type: ignore [prop-decorator]
    @property
    def parent(self) -> "AnyNode | None":
        """Return the parent of this node."""
        return self._parent

    @parent.setter
    def parent(self, value: "AnyNode | None") -> None:
        """Set the parent of this node."""
        if value is not None and self not in value._children:
            value._children.append(cast("AnyNode", self))
        prev, self._parent = self._parent, value
        if value != prev:
            self.events.parent.emit(value, prev)

    @model_serializer(mode="wrap")
    def _serialize_withnode_type(self, handler: SerializerFunctionWrapHandler) -> Any:
        # modified serializer that ensures node_type is included,
        # (e.g. even if exclude_defaults=True)
        data = handler(self)
        if node_type := getattr(self, "node_type", None):
            data["node_type"] = node_type
        return data

    def __contains__(self, item: object) -> bool:
        """Return True if this node is an ancestor of item."""
        return item in self.children

    # below borrowed from vispy.scene.Node

    def transform_to_node(self, other: "Node") -> Transform:
        """Return Transform that maps from coordinate frame of `self` to `other`.

        Note that there must be a _single_ path in the scenegraph that connects
        the two entities; otherwise an exception will be raised.

        Parameters
        ----------
        other : instance of Node
            The other node.

        Returns
        -------
        transform : instance of ChainTransform
            The transform.
        """
        a, b = self.path_to_node(other)
        tforms = [n.transform for n in a[:-1]] + [n.transform.inv() for n in b]
        return Transform.chain(*tforms[::-1])

    def path_to_node(self, other: "Node") -> tuple[list["Node"], list["Node"]]:
        """Return two lists describing the path from this node to another.

        Parameters
        ----------
        other : instance of Node
            The other node.

        Returns
        -------
        p1 : list
            First path (see below).
        p2 : list
            Second path (see below).

        Notes
        -----
        The first list starts with this node and ends with the common parent
        between the endpoint nodes. The second list contains the remainder of
        the path from the common parent to the specified ending node.

        For example, consider the following scenegraph::

            A --- B --- C --- D
                   \
                    --- E --- F

        Calling `D.node_path(F)` will return::

            ([D, C, B], [E, F])

        """
        my_parents = list(self.iter_parents())
        their_parents = list(other.iter_parents())
        common_parent = next((p for p in my_parents if p in their_parents), None)
        if common_parent is None:
            slf = f"{self.__class__.__name__} {id(self)}"
            nd = f"{other.__class__.__name__} {id(other)}"
            raise RuntimeError(f"No common parent between nodes {slf} and {nd}.")

        up = my_parents[: my_parents.index(common_parent) + 1]
        down = their_parents[: their_parents.index(common_parent)][::-1]
        return (up, down)

    def iter_parents(self) -> Iterator["Node"]:
        """Return list of parents starting from this node.

        The chain ends at the first node with no parents.
        """
        yield self

        x = cast("AnyNode", self)
        while True:
            try:
                parent = x.parent
            except Exception:
                break
            if parent is None:
                break
            yield parent
            x = parent

    def tree_repr(self) -> str:
        """Return an ASCII/Unicode tree representation of self and its descendants."""
        from scenex.util import tree_repr

        return tree_repr(self, node_repr=object.__repr__)
