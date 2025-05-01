"""Adaptors are the link between `scenex` models and graphics libraries.

For each model in [`scenex.model`][], there is a corresponding adaptor in
[`scenex.adaptors`][].  "Showing" a model means creating an adaptor for each object
in the model.
"""

from ._auto import get_adaptor_registry
from ._base import Adaptor
from ._registry import AdaptorRegistry

__all__ = ["Adaptor", "AdaptorRegistry", "get_adaptor_registry"]
