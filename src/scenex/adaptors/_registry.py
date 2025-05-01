"""Registry of adaptor objects."""

from __future__ import annotations

import contextlib
import logging
import sys
from functools import cache
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from scenex import model as models

from . import _base

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scenex import model
    from scenex.model._base import EventedBase

_M = TypeVar("_M", bound="model.EventedBase")
logger = logging.getLogger("scenex.adaptors")


class AdaptorRegistry:
    """Weak registry for Adaptor objects.

    Each backend should subclass this and implement the `get_adaptor_class` method.
    And expose an instance of the subclass as `adaptors` in the top level of the backend
    module.
    """

    def __init__(self) -> None:
        self._objects: dict[str, _base.Adaptor] = {}

    def all(self) -> Iterator[_base.Adaptor]:
        """Return an iterator over all adaptors in the registry."""
        yield from self._objects.values()

    # TODO: see if this can be done better with typevars.
    # (it doesn't appear to be trivial)
    @overload
    def get_adaptor(
        self,
        obj: model.Points,
        create: bool = ...,
    ) -> _base.PointsAdaptor: ...
    @overload
    def get_adaptor(
        self,
        obj: model.Image,
        create: bool = ...,
    ) -> _base.ImageAdaptor: ...
    @overload
    def get_adaptor(
        self,
        obj: model.Camera,
        create: bool = ...,
    ) -> _base.CameraAdaptor: ...
    @overload
    def get_adaptor(
        self,
        obj: model.Scene,
        create: bool = ...,
    ) -> _base.NodeAdaptor: ...
    @overload
    def get_adaptor(
        self,
        obj: model.View,
        create: bool = ...,
    ) -> _base.ViewAdaptor: ...
    @overload
    def get_adaptor(
        self,
        obj: model.Canvas,
        create: bool = ...,
    ) -> _base.CanvasAdaptor: ...
    @overload
    def get_adaptor(
        self,
        obj: model.EventedBase,
        create: bool = ...,
    ) -> _base.Adaptor: ...
    def get_adaptor(self, obj: _M, create: bool = True) -> _base.Adaptor[_M, Any]:
        """Get the adaptor for the given model object, create if `create` is True."""
        if obj._model_id.hex not in self._objects:
            if not create:
                raise KeyError(f"No adaptor found for {obj!r}, and create=False")
            logger.debug(
                "Creating %r Adaptor %-14r id: %s",
                type(self).__module__,
                type(obj).__name__,
                obj._model_id.hex[:8],
            )
            self._objects[obj._model_id.hex] = adaptor = self.create_adaptor(obj)
            self.initialize_adaptor(obj, adaptor)
        return self._objects[obj._model_id.hex]

    def initialize_adaptor(
        self, model: model.EventedBase, adaptor: _base.Adaptor
    ) -> None:
        """Initialize the adaptor for the given model object."""
        # syncronize all model properties with the adaptor
        sync_adaptor(adaptor, model)
        # connect the model events to the adaptor, to keep the adaptor in sync

        model.events.connect(adaptor.handle_event)

        if isinstance(model, models.Canvas):
            for view in model.views:
                self.get_adaptor(view, create=True)
        if isinstance(model, models.View):
            self.get_adaptor(model.scene, create=True)
        if isinstance(model, models.Node):
            adaptor = cast("_base.NodeAdaptor", adaptor)
            model.child_added.connect(adaptor._snx_add_child)
            model.child_removed.connect(adaptor._snx_remove_child)
            for child in model.children:
                # perhaps optional ... since _implementations of _snx_add_child
                # will also likely need to call get_adaptor
                self.get_adaptor(child, create=True)
                adaptor._snx_add_child(child)

    def get_adaptor_class(self, obj: model.EventedBase) -> type[_base.Adaptor]:
        """Return the adaptor class for the given model object."""
        cls = type(self)
        cls_module = sys.modules[cls.__module__]
        cls_file = cls_module.__file__
        raise NotImplementedError(
            f"{cls.__name__}.get_adaptor_class not implemented in {cls_file}"
        )

    @classmethod
    def validate_adaptor_class(
        cls, obj: model.EventedBase, adaptor_cls: type[_base.Adaptor]
    ) -> None:
        """Validate that the given class is a valid adaptor for the given object."""
        return _validate_adaptor_class(type(obj), adaptor_cls)

    def create_adaptor(self, model: _M) -> _base.Adaptor[_M, Any]:
        """Create a new adaptor for the given model object."""
        adaptor_cls: type[_base.Adaptor] = self.get_adaptor_class(model)
        self.validate_adaptor_class(model, adaptor_cls)
        adaptor = adaptor_cls(model)

        return adaptor


def _update_blocker(adaptor: _base.Adaptor) -> contextlib.AbstractContextManager:
    if isinstance(adaptor, _base.NodeAdaptor):

        @contextlib.contextmanager
        def blocker() -> Iterator[None]:
            adaptor._snx_block_updates()
            try:
                yield
            finally:
                adaptor._snx_unblock_updates()

        return blocker()
    return contextlib.nullcontext()


def sync_adaptor(adaptor: _base.Adaptor, model: EventedBase) -> None:
    """Decorator to validate and cache adaptor classes."""
    with _update_blocker(adaptor):
        fields = type(model).model_fields | type(model).model_computed_fields
        logger.debug("Synchronizing fields %r", set(fields))
        for field_name in fields:
            method_name = adaptor.SETTER_METHOD.format(name=field_name)
            value = getattr(model, field_name)
            try:
                vis_set = getattr(adaptor, method_name)
                vis_set(value)
            except Exception as e:
                # TODO:
                # it's probably fine that certain snx_set_* methods are not implemented
                # in fact... we may want to flip it around and only get fields based
                # on the available _snx_set_* methods
                logger.debug("MISSING %r: %s", adaptor, e)
    force_update = getattr(adaptor, "_snx_force_update", lambda: None)
    force_update()


@cache
def _validate_adaptor_class(
    obj_type: type[_M], adaptor_cls: type[_base.Adaptor]
) -> None:
    if not isinstance(adaptor_cls, type) and issubclass(adaptor_cls, _base.Adaptor):
        raise TypeError(f"Expected an Adaptor class, got {adaptor_cls!r}")
    # TODO
