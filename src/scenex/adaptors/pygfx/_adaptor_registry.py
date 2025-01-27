from typing import Any

from scenex.adaptors.registry import AdaptorRegistry


class PygfxAdaptorRegistry(AdaptorRegistry):
    def get_adaptor_class(self, obj: Any) -> type:
        from scenex.adaptors import pygfx

        obj_type_name = obj.__class__.__name__
        return getattr(pygfx, f"{obj_type_name}")  # type: ignore


adaptors = PygfxAdaptorRegistry()
get_adaptor = adaptors.get_adaptor
