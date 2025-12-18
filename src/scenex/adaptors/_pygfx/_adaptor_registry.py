from typing import Any

from scenex.adaptors._registry import AdaptorRegistry


class PygfxAdaptorRegistry(AdaptorRegistry):
    def get_adaptor_class(self, obj: Any) -> type:
        from scenex.adaptors import _pygfx

        obj_type_name = obj.__class__.__name__
        return getattr(_pygfx, f"{obj_type_name}")  # type: ignore


adaptors = PygfxAdaptorRegistry()
get_adaptor = adaptors.get_adaptor
