import importlib.util
import os

from ._registry import AdaptorRegistry

CANVAS_ENV_VAR = "SCENEX_CANVAS_BACKEND"


def get_adaptor_registry(backend: str | None = None) -> AdaptorRegistry:
    """Get the backend adaptor registry."""
    if determine_backend(backend) == "vispy":
        from ._vispy import adaptors

        return adaptors
    else:
        from ._pygfx import adaptors  # type: ignore

        return adaptors


def determine_backend(backend: str | None = None) -> str:
    """Get the backend adaptor registry."""
    # Load backend explicitly requested by user
    if backend == "pygfx":
        return "pygfx"
    if backend == "vispy":
        return "vispy"

    # Load backend requested via environment variables
    env_request = os.environ.get(CANVAS_ENV_VAR, None)
    if env_request == "pygfx":
        return "pygfx"
    if env_request == "vispy":
        return "vispy"

    # If no backend is specified, try to find one
    if importlib.util.find_spec("pygfx") is not None:
        return "pygfx"
    if importlib.util.find_spec("vispy") is not None:
        return "vispy"

    for_backend = f" for backend {backend!r}" if backend else ""
    raise RuntimeError(f"No provider found{for_backend} :(")
