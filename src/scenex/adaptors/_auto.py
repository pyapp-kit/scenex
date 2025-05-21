import importlib.util
import os

from ._registry import AdaptorRegistry


def get_adaptor_registry(backend: str | None = None) -> AdaptorRegistry:
    """Get the backend adaptor registry."""
    # Load backend explicitly requested by user
    if backend == "vispy":
        return _load_vispy_adaptors()
    if backend == "pygfx":
        return _load_pygfx_adaptors()

    # Load backend requested via environment variables
    env_request = os.environ.get("SCENEX_CANVAS_BACKEND", None)
    if env_request == "vispy":
        return _load_vispy_adaptors()
    if env_request == "pygfx":
        return _load_pygfx_adaptors()

    # If no backend is specified, try to find one
    if importlib.util.find_spec("vispy") is not None:
        return _load_vispy_adaptors()
    if importlib.util.find_spec("pygfx") is not None:
        return _load_pygfx_adaptors()

    for_backend = f" for backend {backend!r}" if backend else ""
    raise RuntimeError(f"No provider found{for_backend} :(")


def _load_pygfx_adaptors() -> AdaptorRegistry:
    from ._pygfx import adaptors

    return adaptors


def _load_vispy_adaptors() -> AdaptorRegistry:
    from ._vispy import adaptors

    return adaptors
