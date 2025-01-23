from importlib import import_module
from typing import Any

from scenex.backends.adaptor_base import AdaptorRegistry


class PygfxAdaptorRegistry(AdaptorRegistry):
    def get_adaptor_class(self, obj: Any) -> type:
        obj_type_name = obj.__class__.__name__
        module = import_module("scenex.backends.pygfx")
        return getattr(module, f"{obj_type_name}")  # type: ignore


adaptors = PygfxAdaptorRegistry()
get_adaptor = adaptors.get_adaptor
