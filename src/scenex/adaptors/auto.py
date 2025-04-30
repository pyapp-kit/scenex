import importlib.util

from .registry import AdaptorRegistry


def get_adaptor_registry(backend: str | None = None) -> AdaptorRegistry:
    """Get the backend adaptor registry."""
    if backend == "vispy":
        from .vispy import adaptors

        return adaptors
    if backend == "pygfx":
        from .pygfx import adaptors

        return adaptors

    # If no backend is specified, try to find one
    if importlib.util.find_spec("vispy") is not None:
        from .vispy import adaptors

        return adaptors
    if importlib.util.find_spec("pygfx") is not None:
        from .pygfx import adaptors

        return adaptors

    for_backend = f" for backend {backend!r}" if backend else ""
    raise RuntimeError(f"No provider found{for_backend} :(")
