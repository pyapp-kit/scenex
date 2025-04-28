from collections.abc import Iterable
from typing import Any, ClassVar, TypedDict

from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field
from typing_extensions import Unpack


class NodeKwargs(TypedDict, total=False):
    """TypedDict for Node kwargs."""

    name: str | None


class Node(BaseModel):
    """Base class for all nodes."""

    name: str | None = None

    _parent: "Node | None" = PrivateAttr(default=None)
    _children: list["Node"] = PrivateAttr(default_factory=list)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        validate_default=True,
        validate_assignment=True,
    )

    # ------------ custom initialisation -------------------------------------

    def __init__(
        self,
        *,
        children: Iterable["Node | dict[str, Any]"] = (),
        **data: Unpack[NodeKwargs],
    ) -> None:
        super().__init__(**data)

        for ch in children:
            if not isinstance(ch, Node):
                ch = Node.model_validate(ch)
                self.add_child(ch)

    @computed_field  # type: ignore [prop-decorator]
    @property
    def children(self) -> tuple["Node", ...]:
        """Return a tuple of the children of this node."""
        return tuple(self._children)

    def add_child(self, child: "Node") -> None:
        """Add a child node to this node."""
        self._children.append(child)
        child._parent = self

    @computed_field  # type: ignore [prop-decorator]
    @property
    def parent(self) -> "Node | None":
        """Return the parent of this node."""
        return self._parent

    @parent.setter
    def parent(self, value: "Node | None") -> None:
        """Set the parent of this node."""
        if value is not None and self not in value._children:
            value._children.append(self)
        self._parent = value


Node()
