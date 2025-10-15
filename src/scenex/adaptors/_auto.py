from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeGuard, cast, get_args

from scenex.app import app

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scenex.adaptors._base import Adaptor

    from ._registry import AdaptorRegistry

KnownBackend: TypeAlias = Literal["vispy", "pygfx"]
KNOWN_BACKENDS: set[str] = set(get_args(KnownBackend))
_USE: KnownBackend | None = None
CANVAS_ENV_VAR = "SCENEX_CANVAS_BACKEND"


def _ensure_valid_backend(backend: str) -> TypeGuard[KnownBackend]:
    if backend is not None and backend not in KNOWN_BACKENDS:
        raise ValueError(
            f"Invalid backend {backend!r}, must be one of {KNOWN_BACKENDS}."
        )
    return True


def get_adaptor_registry(backend: KnownBackend | str | None = None) -> AdaptorRegistry:
    """Get the backend adaptor registry."""
    match determine_backend(backend):
        case "vispy":
            from . import _vispy

            return _vispy.adaptors
        case "pygfx":
            from . import _pygfx

            return _pygfx.adaptors


def get_all_adaptors(obj: Any) -> Iterator[Adaptor]:
    """Get all adaptors for the given object."""
    for mod_name in ["scenex.adaptors._vispy", "scenex.adaptors._pygfx"]:
        if mod := sys.modules.get(mod_name):
            reg = cast("AdaptorRegistry", mod.adaptors)
            with suppress(KeyError):
                yield reg.get_adaptor(obj, create=False)


def determine_backend(request: KnownBackend | str | None = None) -> KnownBackend:
    """Get the name of the backend adaptor registry.

    In order of priority:
    1. The backend passed to this function.
    2. The backend specified by the use() function.
    3. The backend specified by the SCENEX_CANVAS_BACKEND environment variable.
    4. Auto-determined backend, checking in order: pygfx, vispy.
    """
    requested = request or _USE or os.getenv(CANVAS_ENV_VAR, "").lower() or None
    if requested and _ensure_valid_backend(requested):
        return requested

    # If no backend is specified, try to find one
    if importlib.util.find_spec("pygfx") is not None:
        return "pygfx"
    if importlib.util.find_spec("vispy") is not None:
        return "vispy"

    raise RuntimeError(
        "Could not find a suitable graphics backend. "
        f"Please install one of: {KNOWN_BACKENDS}."
    )


def use(backend: KnownBackend | None = None) -> None:
    """Set the graphics backend, or `None` to auto-determine."""
    global _USE
    if backend is None or _ensure_valid_backend(backend):
        _USE = backend


def run() -> None:
    """Enter the native GUI event loop."""
    app().run()
