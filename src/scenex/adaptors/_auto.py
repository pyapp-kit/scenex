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
    """Get all loaded adaptors for the given object."""
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
    """Set the graphics backend for rendering scenex visualizations.

    This function allows you to explicitly select which graphics library (backend)
    scenex should use for rendering. It is the goal of scenex to support the full range
    of model API for each backend.

    If not called, scenex will automatically select an arbitrary available backend. You
    can also set the backend via the SCENEX_CANVAS_BACKEND environment variable.

    Parameters
    ----------
    backend : Literal["pygfx", "vispy"] | None
        The graphics backend to use:
        - "pygfx": Modern WebGPU-based renderer with advanced features
        - "vispy": OpenGL-based renderer with broad compatibility
        - None: Reset to auto-detection

    Raises
    ------
    ValueError
        If the specified backend is not one of the known backends.

    Examples
    --------
    Use pygfx backend explicitly:
        >>> import scenex as snx
        >>> snx.use("pygfx")  # doctest: +SKIP
        >>> canvas = snx.show(snx.View())

    Use vispy backend:
        >>> snx.use("vispy")  # doctest: +SKIP
        >>> canvas = snx.show(snx.Scene())

    Reset to auto-detection:
        >>> snx.use(None)

    Notes
    -----
    The backend selection follows this priority order:
    1. Backend specified by this function
    2. SCENEX_CANVAS_BACKEND environment variable
    3. Auto-detection (pygfx preferred, then vispy)

    This function should be called before creating any visualizations.
    """
    global _USE
    if backend is None or _ensure_valid_backend(backend):
        _USE = backend


def run() -> None:
    """Start the GUI event loop to display interactive visualizations.

    This function enters the native event loop of the graphics backend, allowing
    interactive visualizations to respond to user input (mouse, keyboard) and remain
    visible. The function blocks until the visualization window is closed.

    Call this function after creating and showing your visualizations with `show()`.
    It is only needed for desktop applications; in Jupyter notebooks, visualizations
    are displayed automatically without calling `run()`.

    Examples
    --------
    Basic usage with a scene:
        >>> import numpy as np
        >>> import scenex as snx
        >>> scene = snx.Scene(
        ...     children=[snx.Image(data=np.random.rand(100, 100).astype(np.float32))]
        ... )
        >>> snx.show(scene)
        Canvas(...)
        >>> snx.run()  # Blocks until window is closed

    Create multiple views and run:
        >>> canvas = snx.Canvas(views=[snx.View(), snx.View()])
        >>> canvas.visible = True
        >>> snx.run()

    Notes
    -----
    - This function blocks execution until all visualization windows are closed
    - Not needed in Jupyter notebooks or other interactive environments
    - Must be called after `show()` has been used to create visualizations
    - The event loop handles user interactions like pan, zoom, and picking
    """
    app().run()
