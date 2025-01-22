import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import suppress
from typing import Any, ClassVar, Generic, TypeVar
from weakref import WeakValueDictionary

from psygnal import SignalGroupDescriptor
from pydantic import BaseModel, ConfigDict, PrivateAttr

_EM = TypeVar("_EM", covariant=True, bound="EventedModel")


class ExtendedConfig(ConfigDict, total=False):
    repr_exclude_defaults: bool


class _ObjectRegistry:
    """Weak registry for all evented model instances."""

    def __init__(self) -> None:
        self._objects: WeakValueDictionary[str, EventedModel] = WeakValueDictionary()

    def register(self, obj: "EventedModel") -> None:
        self._objects[obj._model_id.hex] = obj

    def all(self) -> Iterator["EventedModel"]:
        return self._objects.values()


objects = _ObjectRegistry()


class EventedModel(BaseModel):
    """Base class for all evented pydantic-style models."""

    _model_id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid1)

    events: ClassVar[SignalGroupDescriptor] = SignalGroupDescriptor()
    model_config: ClassVar[ConfigDict] = ExtendedConfig(
        extra="forbid",
        validate_default=True,
        validate_assignment=True,
        repr_exclude_defaults=True,
    )

    def model_post_init(self, __context: Any) -> None:
        """Called after the model is initialized."""
        objects.register(self)

    def __repr_args__(self) -> Iterable[tuple[str | None, Any]]:
        # repr that excludes default values
        super_args = super().__repr_args__()
        if not self.model_config.get("repr_exclude_defaults"):
            yield from super_args
            return

        fields = self.model_fields
        for key, val in super_args:
            if key in fields:
                default = fields[key].get_default(
                    call_default_factory=True, validated_data={}
                )
                with suppress(Exception):
                    if val == default:
                        continue
            yield key, val


class ModelController(ABC, Generic[_EM]):
    """Protocol for backend adaptor classes.

    A controller converts model fields into into native calls for the given backend.
    """

    @abstractmethod
    def __init__(self, obj: _EM) -> None:
        """All backend adaptor objects receive the object they are adapting."""

    @abstractmethod
    def _vis_get_native(self) -> Any:
        """Return the native object for the backend."""


class SupportsVisibility(ModelController[_EM]):
    """Protocol for objects that support visibility (show/hide)."""

    @abstractmethod
    def _vis_set_visible(self, arg: bool) -> None:
        """Set the visibility of the object."""
