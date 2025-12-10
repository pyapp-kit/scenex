from __future__ import annotations

import importlib
import os
import sys
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum, auto
from functools import cache, wraps
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


class CursorType(Enum):
    """Enumeration of standard cursor types for canvas interaction.

    CursorType provides platform-independent cursor shapes that can be set on
    canvases to indicate different interaction modes or states. Each cursor type
    is mapped to the appropriate platform-specific cursor by the GUI backend.

    Attributes
    ----------
    DEFAULT : int
        The standard arrow cursor, typically used for normal selection and interaction.
    CROSS : int
        A crosshair cursor, useful for precise positioning or drawing operations.
    V_ARROW : int
        A vertical resize arrow cursor, indicating vertical resizing capability.
    H_ARROW : int
        A horizontal resize arrow cursor, indicating horizontal resizing capability.
    ALL_ARROW : int
        A multi-directional arrow cursor, indicating omnidirectional movement.
    BDIAG_ARROW : int
        A diagonal resize arrow cursor (backward diagonal), for diagonal resizing.
    FDIAG_ARROW : int
        A diagonal resize arrow cursor (forward diagonal), for diagonal resizing.

    Examples
    --------
    Set a crosshair cursor during drawing mode:
        >>> app().set_cursor(canvas, CursorType.CROSS)

    Restore default cursor after operation:
        >>> app().set_cursor(canvas, CursorType.DEFAULT)

    See Also
    --------
    App.set_cursor : Method to set cursor on a canvas
    """

    DEFAULT = auto()
    CROSS = auto()
    V_ARROW = auto()
    H_ARROW = auto()
    ALL_ARROW = auto()
    BDIAG_ARROW = auto()
    FDIAG_ARROW = auto()


class App:
    """Base class for GUI application wrappers.

    App provides an abstract interface for integrating scenex with different GUI
    frameworks (Qt, WxPython, Jupyter). Each GUI backend implements this interface
    to provide framework-specific application lifecycle management, event handling,
    and threading operations.

    The App class is typically accessed via the `app()` function, which automatically
    determines and initializes the appropriate backend based on the environment and
    available GUI frameworks.

    Notes
    -----
    This is an abstract base class. Concrete implementations are provided by
    backend-specific subclasses (QtAppWrap, WxAppWrap, JupyterAppWrap).

    See Also
    --------
    app : Function to get the active application instance
    GuiFrontend : Enumeration of available GUI backends
    determine_app : Function to determine which GUI backend to use
    """

    def create_app(self) -> Any:
        """Create the application instance, if not already created.

        This method initializes the underlying GUI framework's application object
        (e.g., QApplication for Qt). If an application instance already exists,
        this method should return the existing instance.

        Returns
        -------
        Any
            The backend-specific application object.

        Notes
        -----
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented by subclasses.")

    def run(self) -> None:
        """Start the application event loop.

        This method enters the GUI framework's main event loop, which processes
        user input, window events, and other GUI operations. The method blocks
        until the application is closed.

        Notes
        -----
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented by subclasses.")

    def show(self, canvas: CanvasAdaptor, visible: bool) -> None:
        """Show or hide a canvas window.

        Parameters
        ----------
        canvas : CanvasAdaptor
            The canvas adaptor wrapping the backend-specific canvas widget.
        visible : bool
            True to show the canvas window, False to hide it.

        Notes
        -----
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented by subclasses.")

    def install_event_filter(self, canvas: Any, model_canvas: Canvas) -> EventFilter:
        """Install an event filter on a canvas to forward events to the model.

        Implementations of this method will capture all events given to the native
        widget, translated them into scenex events, and route them to `model_canvas`.

        Parameters
        ----------
        canvas : Any
            The backend-specific native canvas widget.
        model_canvas : Canvas
            The scenex Canvas model that should receive events.

        Returns
        -------
        EventFilter
            A handle that can be used to uninstall the event filter.

        Notes
        -----
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented by subclasses.")

    def process_events(self) -> None:
        """Yields the current thread to process all pending GUI events.

        Notes
        -----
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented by subclasses.")

    def call_in_main_thread(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        """Schedule a function to be called in the main GUI thread.

        Many GUI frameworks require that widget operations occur on the main
        thread. This method safely schedules a function call on the main thread
        and returns a Future that will contain the result.

        Parameters
        ----------
        func : Callable[P, T]
            The function to call.
        *args : P.args
            Positional arguments to pass to func.
        **kwargs : P.kwargs
            Keyword arguments to pass to func.

        Returns
        -------
        Future[T]
            A Future object that will contain the function's return value once
            the call completes.

        Notes
        -----
        The base implementation executes the function immediately and returns
        a completed Future. Subclasses should override this to provide
        thread-safe execution.
        """
        future: Future[T] = Future()
        future.set_result(func(*args, **kwargs))
        return future

    def call_later(self, msec: int, func: Callable[[], None]) -> None:
        """Schedule a function to be called after a delay.

        Parameters
        ----------
        msec : int
            Delay in milliseconds before calling the function.
        func : Callable[[], None]
            The function to call. Must take no arguments.

        Notes
        -----
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented by subclasses.")

    def get_executor(self) -> Executor:
        """Return an executor for running tasks in background threads.

        Returns
        -------
        Executor
            A concurrent.futures.Executor instance (typically a ThreadPoolExecutor)
            for running background tasks.

        Notes
        -----
        The default implementation returns a shared ThreadPoolExecutor with 2
        workers. Subclasses can override this to provide framework-specific
        executors.
        """
        return _thread_pool_executor()

    @contextmanager
    def block_events(self, window: Any) -> Iterator[None]:
        """Context manager to temporarily block events for a window.

        Parameters
        ----------
        window : Any
            The backend-specific window object.

        Yields
        ------
        None

        Notes
        -----
        Must be implemented by subclasses.

        Examples
        --------
        Block events during a long operation:
            >>> with app().block_events(canvas_widget):
            ...     perform_long_operation()
        """
        raise NotImplementedError("Must be implemented by subclasses.")

    # ------------------------------ cursor API -------------------------------
    def set_cursor(self, canvas: Canvas, cursor: CursorType) -> None:
        """Set the cursor for the given canvas.

        Backends override this to translate the abstract cursor into native form.

        Parameters
        ----------
        canvas : Canvas
            The canvas on which to set the cursor.
        cursor : CursorType
            The type of cursor to set.
        """
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


def ensure_main_thread(func: Callable[P, T]) -> Callable[P, Future[T]]:
    """Decorator that ensures a function is called in the main GUI thread.

    This decorator wraps a function so that when called, it is automatically
    scheduled to run on the main GUI thread rather than the caller's thread.
    This is essential for GUI operations that must occur on the main thread.

    Parameters
    ----------
    func : Callable[P, T]
        The function to wrap. Can have any signature.

    Returns
    -------
    Callable[P, Future[T]]
        A wrapped version of func that returns a Future instead of the direct
        result. The Future will contain the function's return value once the
        call completes on the main thread.

    Examples
    --------
    Ensure a GUI operation runs on the main thread:
        >>> @ensure_main_thread
        ... def update_widget(value: int) -> None:
        ...     widget.set_value(value)
        >>> future = update_widget(42)  # Returns immediately with Future
        >>> result = future.result()  # Block until completion if needed

    See Also
    --------
    App.call_in_main_thread : Underlying method for thread-safe calls
    """

    @wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> Future[T]:
        return app().call_in_main_thread(func, *args, **kwargs)

    return _wrapper


def determine_app() -> GuiFrontend:
    """Determine which GUI backend to use for the application.

    This function selects the appropriate GUI framework backend through a
    three-tier strategy:

    1. **Explicit request**: If the SCENEX_WIDGET_BACKEND environment variable
       is set, that backend is used (e.g., "qt", "wx", "jupyter").
    2. **Running application**: If a GUI application is already running in the
       process (detected via framework imports), that backend is used.
    3. **Available backend**: Try importing each backend in a predefined order until one
       succeeds.

    Returns
    -------
    GuiFrontend
        The determined GUI backend to use.

    Raises
    ------
    ValueError
        If the SCENEX_WIDGET_BACKEND environment variable is set to an invalid
        value.
    RuntimeError
        If no GUI backend can be found or loaded.

    Examples
    --------
    Let the function auto-detect the backend:
        >>> backend = determine_app()

    Force a specific backend via environment variable:
        >>> import os
        >>> os.environ["SCENEX_WIDGET_BACKEND"] = "qt"
        >>> backend = determine_app()  # Will use Qt

    See Also
    --------
    app : Get the active App instance using the determined backend
    GuiFrontend : Enumeration of available backends
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
    """Get the active GUI application instance.

    Returns the singleton App instance for the current process, creating and
    initializing it if necessary. The GUI backend is determined automatically
    using `determine_app()`.

    This function should be used whenever you need to interact with the GUI
    application, such as running the event loop, showing windows, or scheduling
    thread-safe operations.

    Returns
    -------
    App
        The active App instance wrapping the GUI backend.

    Examples
    --------
    Get the app and run the event loop:
        >>> app().run()

    Show a canvas window:
        >>> from scenex import Canvas
        >>> canvas = Canvas()
        >>> app().show(canvas._get_adaptors()[0], visible=True)

    Process pending events without blocking:
        >>> app().process_events()

    See Also
    --------
    determine_app : Function that selects which GUI backend to use
    GuiFrontend : Enumeration of available backends
    App : Base class defining the application interface
    """
    global _APP
    if _APP is not None:
        return _APP

    # ensure the app is created for explicitly requested frontends
    _APP = _load_app(*GUI_PROVIDERS[determine_app()])
    _APP.create_app()
    return _APP
