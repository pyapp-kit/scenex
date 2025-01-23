import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import suppress
from typing import Any, ClassVar, Generic, TypeVar
from weakref import WeakValueDictionary

from psygnal import EmissionInfo, SignalGroupDescriptor
from pydantic import BaseModel, ConfigDict, PrivateAttr

_EM = TypeVar("_EM", covariant=True, bound="EventedBase")


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


_AT = TypeVar("_AT")
logger = logging.getLogger(__name__)


class Adaptor(ABC, Generic[_EM, _AT]):
    """ABC for backend adaptor classes.

    An adaptor converts model change events into into native calls for the given
    backend.
    """

    SETTER_METHOD = "_snx_set_{name}"

    @abstractmethod
    def __init__(self, obj: _EM) -> None:
        """All backend adaptor objects receive the object they are adapting."""

    @abstractmethod
    def _snx_get_native(self) -> _AT:
        """Return the native object for the ."""

    def handle_event(self, info: EmissionInfo) -> None:
        signal_name = info.signal.name

        try:
            name = self.SETTER_METHOD.format(name=signal_name)
            setter = getattr(self, name)
        except AttributeError as e:
            logger.exception(e)
            return

        event_name = f"{type(self).__name__}.{signal_name}"
        logger.debug(f"{event_name}={info.args} emitting to backend")

        try:
            setter(info.args[0])
        except Exception as e:
            logger.exception(e)


class SupportsVisibility(Adaptor[_EM, _AT]):
    """Protocol for objects that support visibility (show/hide)."""

    @abstractmethod
    def _snx_set_visible(self, arg: bool) -> None:
        """Set the visibility of the object."""
