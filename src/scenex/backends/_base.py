from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload
from weakref import WeakValueDictionary

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scenex.model import Node, Transform
    from scenex.model._base import EventedModel


_M = TypeVar("_M", bound="EventedModel")
_MCo = TypeVar("_MCo", covariant=True, bound="EventedModel")
_N = TypeVar("_N")


class Adaptor(ABC, Generic[_MCo, _N]):
    """Protocol for backend adaptor classes.

    A controller converts model fields into into native calls for the given backend.
    """

    def __init__(self, obj: _MCo) -> None:
        """All backend adaptor objects receive the object they are adapting."""
        self.model = obj
        self.post_init()

    def post_init(self) -> None:
        """Called after the model is initialized."""
        pass

    @abstractmethod
    def _vis_get_native(self) -> _N:
        """Return the native object for the backend."""


class SupportsVisibility(Adaptor[_MCo, _N]):
    """Protocol for objects that support visibility (show/hide)."""

    @abstractmethod
    def _vis_set_visible(self, arg: bool) -> None:
        """Set the visibility of the object."""


class NodeAdaptor(SupportsVisibility[_MCo, _N]):
    """Backend interface for a Node."""

    @abstractmethod
    def _vis_set_name(self, arg: str) -> None: ...
    @abstractmethod
    def _vis_set_parent(self, arg: Node | None) -> None: ...
    @abstractmethod
    def _vis_set_children(self, arg: list[Node]) -> None: ...
    @abstractmethod
    def _vis_set_opacity(self, arg: float) -> None: ...
    @abstractmethod
    def _vis_set_order(self, arg: int) -> None: ...
    @abstractmethod
    def _vis_set_interactive(self, arg: bool) -> None: ...
    @abstractmethod
    def _vis_set_transform(self, arg: Transform) -> None: ...
    @abstractmethod
    def _vis_add_node(self, node: Node) -> None: ...

    @abstractmethod
    def _vis_block_updates(self) -> None:
        """Block future updates until `unblock_updates` is called."""

    @abstractmethod
    def _vis_unblock_updates(self) -> None:
        """Unblock updates after `block_updates` was called."""

    @abstractmethod
    def _vis_force_update(self) -> None:
        """Force an update to the node."""


class _AdaptorRegistry:
    """Weak registry for all evented model instances."""

    def __init__(self) -> None:
        self._objects: WeakValueDictionary[str, Adaptor] = WeakValueDictionary()

    def all(self) -> Iterator[Adaptor]:
        return self._objects.values()

    def get_adaptor(self, obj: _M) -> Adaptor[_M, Any]:
        if obj._model_id not in self._objects:
            self._objects[obj._model_id.hex] = self.create_adaptor(obj)
        return self._objects[obj._model_id.hex]

    @overload
    def create_adaptor(self, obj: Node) -> NodeAdaptor: ...
    @overload
    def create_adaptor(self, obj: EventedModel) -> Adaptor: ...
    def create_adaptor(self, obj: _M) -> Adaptor[_M, Any]:
        raise NotImplementedError("Subclasses must implement this method.")


adaptors = _AdaptorRegistry()
