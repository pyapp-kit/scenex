import uuid
from collections.abc import Iterable, Iterator
from contextlib import suppress
from typing import Any, ClassVar
from weakref import WeakValueDictionary

from psygnal import SignalGroupDescriptor
from pydantic import BaseModel, ConfigDict, PrivateAttr


class ExtendedConfig(ConfigDict, total=False):
    repr_exclude_defaults: bool


class _ObjectRegistry:
    """Weak registry for all evented model instances."""

    def __init__(self) -> None:
        self._objects: WeakValueDictionary[str, EventedBase] = WeakValueDictionary()

    def register(self, obj: "EventedBase") -> None:
        self._objects[obj._model_id.hex] = obj

    def all(self) -> Iterator["EventedBase"]:
        return self._objects.values()


objects = _ObjectRegistry()


class EventedBase(BaseModel):
    """Base class for all evented pydantic-style models."""

    _model_id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid1)

    events: ClassVar[SignalGroupDescriptor] = SignalGroupDescriptor()

    # note, strangely, for mypy reasons,
    # this configDict should not include extra="forbid"
    # it's a long story: https://github.com/pydantic/pydantic/issues/11329
    model_config: ClassVar[ConfigDict] = ExtendedConfig(
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
