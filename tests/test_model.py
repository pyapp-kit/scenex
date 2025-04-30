import sys

import pytest
from typing_extensions import TypeAliasType

from scenex import model
from scenex.adaptors.base import Adaptor
from scenex.model._base import EventedBase


# recursively collect all subclasses of a type
def collect_adaptors(cls: type) -> list[type]:
    """Recursively collect all subclasses of Adaptor."""
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(collect_adaptors(subclass))
    return subclasses


ADAPTORS = collect_adaptors(Adaptor)


def _get_model_type(cls: TypeAliasType) -> type[EventedBase]:
    """Get the model class for a given adaptor class."""
    params = cls.__parameters__
    if not (ref := params[0].__bound__):
        raise ValueError(
            f"Cannot get model class for {cls.__name__}: no bound type found"
        )
    if sys.version_info >= (3, 12):
        model_type = ref._evaluate(
            {"model": model}, None, type_params=None, recursive_guard=frozenset()
        )
    else:
        model_type = ref._evaluate({"model": model}, None, recursive_guard=frozenset())
    assert issubclass(model_type, EventedBase)
    return model_type  # type: ignore [no-any-return]


def test_schema() -> None:
    assert model.Canvas.model_json_schema(mode="serialization")
    assert model.Canvas.model_json_schema(mode="validation")


@pytest.mark.parametrize("adaptor", ADAPTORS)
def test_events(adaptor: TypeAliasType) -> None:
    """Test that models have events corresponding to all _snx_set_* methods."""
    snx_methods = {
        name[9:]
        for name, method in adaptor.__dict__.items()
        if name.startswith("_snx_set_") and callable(method)
    }

    model_type = _get_model_type(adaptor)
    if not model_type.model_fields:
        # no fields, so no events
        return
    signal_group = model_type.events._create_group(model_type)
    signal_names = set(signal_group._psygnal_signals)
    # remove the _snx_set_ prefix from the method names
    # find the difference between the two sets
    missing = snx_methods - signal_names
    if missing:
        raise AssertionError(
            f"Missing events on {model_type.__module__}.{model_type.__name__}: "
            f"{', '.join(sorted(missing))}"
        )
