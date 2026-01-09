from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, TypeAlias, Union, cast

import numpy as np
from psygnal import Signal
from pydantic import (
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    computed_field,
    model_serializer,
    model_validator,
)

from scenex.model._base import EventedBase
from scenex.model._transform import Transform

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import numpy.typing as npt
    from typing_extensions import Self, TypedDict, Unpack

    from scenex.app.events import Ray

    from .camera import Camera
    from .image import Image
    from .line import Line
    from .points import Points
    from .scene import Scene

    class NodeKwargs(TypedDict, total=False):
        """TypedDict for Node kwargs."""

        parent: Node | None
        name: str | None
        visible: bool
        interactive: bool
        opacity: float
        order: int
        transform: Transform | npt.ArrayLike


logger = logging.getLogger(__name__)


# improve me... Read up on: https://docs.pydantic.dev/latest/concepts/unions/
AnyNode = Annotated[
    Union["Image", "Points", "Line", "Camera", "Scene"],
    Field(discriminator="node_type"),
]

# Axis-Aligned Bounding Box
AABB: TypeAlias = tuple[tuple[float, float, float], tuple[float, float, float]]


class BlendMode(Enum):
    """
    A set of available blending modes.

    Blending modes determine how the colors of rendered objects are combined with the
    colors already present in the framebuffer. More practically, if two objects overlap
    from the camera's perspective in the scene, the blending mode of the new object
    determines how its colors are combined with those of the object previously rendered.

    Note that the draw order plays a crucial role in blending.
    """

    OPAQUE = "opaque"
    """The object's color value, multiplied by its alpha value, overwrites the
    background color.
    """
    ALPHA = "alpha"
    """
    The object's color is blended with the background using standard alpha compositing.
    The resulting color is a weighted combination of the foreground and background,
    where weights are determined by alpha values.
    """
    ADDITIVE = "additive"
    """The object's color value, multiplied by its alpha value, is added to the
    background color.
    """


class Node(EventedBase):
    """Base class for all nodes in the scene graph.

    Node is the fundamental building block of scenex's scene graph architecture. Nodes
    form a hierarchical tree structure where each node can have a parent and children,
    creating a parent-child relationship that propagates transformations, visibility,
    and other properties through the graph.

    Nodes are abstract and should not be instantiated directly. Use concrete subclasses
    like Image, Points, Line, Mesh, Scene, or Camera instead.

    The scene graph hierarchy allows:
    - Hierarchical transformations: A node's transform is relative to its parent
    - Property inheritance: Visibility and opacity affect all descendants
    - Spatial relationships: Nodes can find paths to other nodes in the graph
    - Event handling: Interactive nodes can respond to user input

    Attributes
    ----------
    parent : Node | None
        The parent node in the scene graph hierarchy. None for root nodes.
    name : str | None
        Optional name for the node, useful for debugging and identification.
    visible : bool
        Whether this node and its children should be rendered.
    interactive : bool
        Whether this node can receive and respond to mouse and touch events.
    opacity : float
        Opacity of the node, from 0.0 (fully transparent) to 1.0 (fully opaque).
    order : int
        Drawing order within siblings. Higher values are drawn later (on top).
        Children are always drawn after their parents.
    transform : Transform
        Transformation mapping the node's local coordinate frame to its parent's frame.
        Applied hierarchically through the scene graph.
    blending : BlendMode
        How this node's colors blend with nodes behind it (opaque, alpha, or additive).

    Notes
    -----
    Do not instantiate Node directly. Use concrete subclasses instead.
    """

    parent: Node | None = Field(
        default=None,
        repr=False,
        exclude=True,
        description="The parent of this node in the scene graph hierarchy",
    )
    # see computed field below
    _children: list[AnyNode] = PrivateAttr(default_factory=list)

    name: str | None = Field(default=None, description="Name identifying the node")
    visible: bool = Field(
        default=True, description="Whether this node and its children are visible"
    )
    interactive: bool = Field(
        default=False,
        description="Whether this node can receive mouse and touch events",
        repr=False,
    )
    opacity: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Opacity from 0.0 (transparent) to 1.0 (opaque)",
    )
    order: int = Field(
        default=0,
        ge=0,
        description=(
            "Drawing order within siblings; higher values drawn later (on top). "
            "Equivalent drawing order for siblings is undefined."
        ),
    )
    transform: Transform = Field(
        default_factory=Transform,
        description="Transformation from local coordinates to parent coordinates",
    )
    blending: BlendMode = Field(
        default=BlendMode.OPAQUE,
        description="How this node's colors blend with nodes behind it",
    )

    model_config = ConfigDict(extra="forbid")

    child_added: ClassVar[Signal] = Signal(object)
    child_removed: ClassVar[Signal] = Signal(object)

    def __init__(
        self,
        *,
        children: Iterable[Node | dict[str, Any]] = (),
        **data: Unpack[NodeKwargs],
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def children(self) -> tuple[Node, ...]:
        """Return a tuple of the children of this node."""
        return tuple(self._children)

    @computed_field  # type: ignore[prop-decorator]
    @property  # TODO: Cache?
    def bounding_box(self) -> AABB | None:
        bounded_nodes = [c for c in self.children if c.bounding_box]
        if not bounded_nodes:
            # If there are no children declaring a bounding box, return None
            return None
        node_aabbs = [n.transform.map(n.bounding_box)[:, :3] for n in bounded_nodes]  # type:ignore
        mi = np.vstack([t[0] for t in node_aabbs]).min(axis=0)
        ma = np.vstack([t[1] for t in node_aabbs]).max(axis=0)
        # Note the casting is important for pydantic
        # FIXME: Should just validate in pydantic
        return (tuple(float(m) for m in mi), tuple(float(m) for m in ma))  # type: ignore

    def add_child(self, child: AnyNode) -> None:
        """Add a child node to this node."""
        self._children.append(child)
        child.parent = cast("AnyNode", self)
        self.child_added.emit(child)

    def remove_child(self, child: AnyNode) -> None:
        """Remove a child node from this node. Does not raise if child is missing."""
        if child in self._children:
            self._children.remove(child)
            child.parent = None
            self.child_removed.emit(child)

    def passes_through(self, ray: Ray) -> float | None:
        """Returns the depth t at which the provided ray intersects this node.

        The ray, in this case, is defined by R(t) = ray_origin + ray_direction * t,
        where t>=0

        Parameters
        ----------
        ray : Ray
            The ray passing through the scene

        Returns
        -------
        t: float | None
            The depth t at which the ray intersects the node, or None if it never
            intersects.
        """
        # Nodes that want to support ray intersection should override this method.
        return None

    @model_validator(mode="wrap")
    @classmethod
    def _validate_model(
        cls,
        value: Any,
        handler: ModelWrapValidatorHandler[Self],
        info: ValidationInfo,
    ) -> Self:
        # Ensures that changing the parent of a node
        # also updates the children of the new/old parent.
        if isinstance(value, dict):
            old_parent = value.get("parent")
        else:
            old_parent = getattr(value, "parent", None)
        result = handler(value)
        cls._update_parent_children(result, old_parent)
        return result

    @staticmethod
    def _update_parent_children(node: Node, old_parent: Node | None = None) -> None:
        """Remove the node from its old_parent and add it to its new parent."""
        if (new_parent := node.parent) != old_parent:
            if new_parent is not None and node not in new_parent._children:
                new_parent._children.append(cast("AnyNode", node))
                new_parent.child_added.emit(node)
            if old_parent is not None and node in old_parent._children:
                old_parent._children.remove(cast("AnyNode", node))
                old_parent.child_removed.emit(node)

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

    def transform_to_node(self, other: Node) -> Transform:
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

    def path_to_node(self, other: Node) -> tuple[list[Node], list[Node]]:
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

    def iter_parents(self) -> Iterator[Node]:
        """Return list of parents starting from this node.

        The chain ends at the first node with no parents.
        """
        yield self

        x = self
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
