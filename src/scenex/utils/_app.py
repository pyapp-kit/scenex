"""Utility functions pertaining to the underlying application.

The `show()` function is the primary entry point for creating visualizations,
handling the details of canvas creation, backend selection, and camera fitting
automatically for a provided node, view or canvas.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from scenex import model
from scenex.app import app

from . import projections

if TYPE_CHECKING:
    from scenex.adaptors._base import CanvasAdaptor
    from scenex.app._auto import CursorType

logger = logging.getLogger("scenex")


def show(
    obj: model.Node | model.View | model.Canvas, *, backend: str | None = None
) -> model.Canvas:
    """Display a visualization by creating a canvas and making it visible.

    This is the primary function for creating and displaying scenex visualizations.
    It accepts nodes, views, or canvases, automatically wrapping them in the necessary
    container objects and creating the appropriate backend adaptors.

    The function automatically fits the camera view to show all visible content and
    makes the canvas window visible. After calling `show()`, use `run()` to enter
    the event loop (in desktop applications) or continue working (in notebooks).

    Parameters
    ----------
    obj : Node | View | Canvas
        The object to visualize:
        - Node (Image, Points, Line, etc.): Wrapped in Scene and View automatically
        - Scene: Wrapped in a View with a default Camera
        - View: Placed on a new Canvas
        - Canvas: Displayed directly (already contains Views)
    backend : str | None, optional
        Graphics backend to use ("pygfx" or "vispy"). If None, uses the backend
        specified by `use()`, `SCENEX_CANVAS_BACKEND` environment variable, or
        auto-detection. Default is `None`.

    Returns
    -------
    Canvas
        The canvas containing the visualization. Can be used to further manipulate
        the display or access the created views.

    Examples
    --------
    Show a simple image:
        >>> import numpy as np
        >>> import scenex as snx
        >>> data = np.random.rand(100, 100).astype(np.float32)
        >>> img = snx.Image(data=data)
        >>> canvas = snx.show(img)
        >>> snx.run()

    Edit the returned canvas:
        >>> from cmap import Color
        >>> canvas.background_color = Color("white")
        >>> canvas.width = 800
        >>> snx.run()

    Notes
    -----
    - The camera is automatically zoomed to fit all visible content with 90% coverage
    - Canvas size defaults to the view's layout dimensions
    - Call `run()` after `show()` to enter the event loop in desktop applications
    - In Jupyter notebooks, visualizations appear automatically without `run()`
    """
    from scenex.adaptors import get_adaptor_registry

    view = None
    if isinstance(obj, model.Canvas):
        canvas = obj
    else:
        if isinstance(obj, model.View):
            view = obj
        elif isinstance(obj, model.Scene):
            view = model.View(scene=obj)
        elif isinstance(obj, model.Node):
            scene = model.Scene(children=[obj])
            view = model.View(scene=scene)

        canvas = model.Canvas()
        if view:
            canvas.views.append(view)

    canvas.visible = True
    reg = get_adaptor_registry(backend=backend)
    reg.get_adaptor(canvas, create=True)
    app().create_app()
    for view in canvas.views:
        projections.zoom_to_fit(view, zoom_factor=0.9, letterbox=True)
    return canvas


def native(canvas: model.Canvas, create: bool = True) -> Any:
    """Get the native widget for the given canvas.

    Parameters
    ----------
    canvas : model.Canvas
        The canvas for which to get the native widget.
    create : bool, optional
        Whether to create adaptors if they do not already exist. Defaults to `True`.

    Returns
    -------
    Any
        The native widget associated with the canvas.

    Raises
    ------
    KeyError
        If no adaptor yet exists for `canvas` and `create=False`.

    Notes
    -----
    This function is a convenience that retrieves the native widget from the first
    adaptor associated with the canvas. If multiple adaptors are present, it returns the
    native widget from the first one found.
    """
    for adaptor in canvas._get_adaptors(create=create):
        return cast("CanvasAdaptor", adaptor)._snx_get_native()


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


def set_cursor(canvas: model.Canvas, cursor: CursorType) -> None:
    """Set the cursor for the given canvas.

    Parameters
    ----------
    canvas : model.Canvas
        The canvas on which to set the cursor.
    cursor : CursorType
        The type of cursor to set.

    Notes
    -----
    Practically and generally speaking, setting the cursor is an app-level concern.
    Unfortunately, setting the cursor often requires access to a native widget, meaning
    any scenex abstractions for setting the cursor will need as input the canvas model
    or a derivative adaptor. Proper separation of concerns suggests that the app-level
    API should just take the native widget. This function is a convenience that performs
    the intermediate steps to get the native widget from a canvas model.
    """
    for adaptor in canvas._get_adaptors(create=True):
        widget = cast("CanvasAdaptor", adaptor)._snx_get_native()
        app().set_cursor(widget, cursor)
