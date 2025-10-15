from __future__ import annotations

import importlib
import os
import sys
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any

    from typing_extensions import ParamSpec, TypeVar

    from scenex.adaptors._base import CanvasAdaptor
    from scenex.app.events._events import EventFilter
    from scenex.model import Canvas

    T = TypeVar("T")
    P = ParamSpec("P")


GUI_ENV_VAR = "SCENEX_WIDGET_BACKEND"
"""Preferred GUI frontend. If not set, the first available GUI frontend is used."""
_APP: App | None = None
"""Singleton instance of the current (GUI) application. Once set it shouldn't change."""


class GuiFrontend(str, Enum):
    """Enum of available GUI frontends.

    Attributes
    ----------
    JUPYTER : str
        [JUPYTER](https://jupyter.org/)
    QT : str
        [PyQt5/PySide2/PyQt6/PySide6](https://doc.qt.io)
    WX : str
        [WX](https://wxpython.org/)
    """

    JUPYTER = "jupyter"
    QT = "qt"
    WX = "wx"


GUI_PROVIDERS: dict[GuiFrontend, tuple[str, str]] = {
    GuiFrontend.WX: ("scenex.app._wx", "WxAppWrap"),
    GuiFrontend.QT: ("scenex.app._qt", "QtAppWrap"),
    # Note that Jupyter should go last because it is a guess based on IPython
    # which may be installed with the other frameworks as well.
    GuiFrontend.JUPYTER: ("scenex.app._jupyter", "JupyterAppWrap"),
}


class App:
    """
    Base class for application wrappers.

    TODO: Where should this live? Probably doesn't belong in this repo...
    """

    def create_app(self) -> Any:
        """Create the application instance, if not already created."""
        raise NotImplementedError("Must be implemented by subclasses.")

    def run(self) -> None:
        """Run the application."""
        raise NotImplementedError("Must be implemented by subclasses.")

    def show(self, canvas: CanvasAdaptor, visible: bool) -> None:
        """Show or hide the canvas."""
        raise NotImplementedError("Must be implemented by subclasses.")

    def install_event_filter(self, canvas: Any, model_canvas: Canvas) -> EventFilter:
        raise NotImplementedError("Must be implemented by subclasses.")

    def process_events(self) -> None:
        """Process events."""
        raise NotImplementedError("Must be implemented by subclasses.")

    def call_in_main_thread(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        """Call `func` in the main gui thread."""
        future: Future[T] = Future()
        future.set_result(func(*args, **kwargs))
        return future

    def call_later(self, msec: int, func: Callable[[], None]) -> None:
        """Call `func` after `msec` milliseconds."""
        raise NotImplementedError("Must be implemented by subclasses.")

    def get_executor(self) -> Executor:
        """Return an executor for running tasks in the background."""
        return _thread_pool_executor()

    @contextmanager
    def block_events(self, window: Any) -> Iterator[None]:
        """Context manager to block events for a window."""
        raise NotImplementedError("Must be implemented by subclasses.")


@cache
def _thread_pool_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)


def _running_apps() -> Iterator[GuiFrontend]:
    """Return an iterator of running GUI applications."""
    for mod_name in ("PyQt5", "PySide2", "PySide6", "PyQt6"):
        if mod := sys.modules.get(f"{mod_name}.QtWidgets"):
            print(f"Found {mod}")
            if (
                qapp := getattr(mod, "QApplication", None)
            ) and qapp.instance() is not None:
                yield GuiFrontend.QT
    # wx
    if (wx := sys.modules.get("wx")) and wx.App.Get() is not None:
        yield GuiFrontend.WX

    # Jupyter notebook
    if (ipy := sys.modules.get("IPython")) and (shell := ipy.get_ipython()):
        if shell.__class__.__name__ == "ZMQInteractiveShell":
            yield GuiFrontend.JUPYTER


def _load_app(module: str, cls_name: str) -> App:
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    return cast("App", cls())


def determine_app() -> GuiFrontend:
    """Returns the [`GuiFrontend`][scenex.app.GuiFrontend].

    This is determined first by the `NDV_GUI_FRONTEND` environment variable, after which
    known GUI providers are tried in order until one is found that is either already
    running, or available.
    """
    running = list(_running_apps())

    # Try 1: Load a frontend explicitly requested by the user
    requested = os.getenv(GUI_ENV_VAR, "").lower()
    valid = {x.value for x in GuiFrontend}
    if requested:
        if requested not in valid:
            raise ValueError(
                f"Invalid GUI frontend: {requested!r}. Valid options: {valid}"
            )
        return GuiFrontend(requested)

    # Try 2: Utilize an existing, running app
    for key in GUI_PROVIDERS.keys():
        if key in running:
            return key

    # Try 3: Load an existing app
    errors: list[tuple[str, BaseException]] = []
    for key, provider in GUI_PROVIDERS.items():
        try:
            _load_app(*provider)
            return key
        except Exception as e:
            errors.append((key, e))

    raise RuntimeError(  # pragma: no cover
        f"Could not find an appropriate GUI frontend: {valid!r}. Tried:\n\n"
        + "\n".join(f"- {key}: {err}" for key, err in errors)
    )


def app() -> App:
    """Return the active [`GuiFrontend`][ndv.views.GuiFrontend].

    This is determined first by the `NDV_GUI_FRONTEND` environment variable, after which
    known GUI providers are tried in order until one is found that is either already
    running, or available.
    """
    global _APP
    if _APP is not None:
        return _APP

    # ensure the app is created for explicitly requested frontends
    _APP = _load_app(*GUI_PROVIDERS[determine_app()])
    _APP.create_app()
    return _APP
