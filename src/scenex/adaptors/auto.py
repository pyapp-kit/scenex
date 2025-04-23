import importlib.util

from .registry import AdaptorRegistry


def get_adaptor_registry() -> AdaptorRegistry:
    """Get the backend adaptor registry."""
    if importlib.util.find_spec("vispy") is not None:
        from .vispy import adaptors

        return adaptors
    if importlib.util.find_spec("pygfx") is not None:
        from .pygfx import adaptors

        return adaptors
    raise RuntimeError("No provider found :(")
