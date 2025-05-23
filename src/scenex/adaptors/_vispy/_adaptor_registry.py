from typing import Any

from scenex.adaptors._registry import AdaptorRegistry


class VispyAdaptorRegistry(AdaptorRegistry):
    def get_adaptor_class(self, obj: Any) -> type:
        from scenex.adaptors import _vispy

        obj_type_name = obj.__class__.__name__
        return getattr(_vispy, f"{obj_type_name}")  # type: ignore


adaptors = VispyAdaptorRegistry()
get_adaptor = adaptors.get_adaptor
