"""Application and GUI framework abstraction layer.

This module provides a unified interface for managing GUI applications across different
frameworks (Qt, WxPython, Jupyter) and rendering backends. It handles the event loop,
window creation, and platform-specific details, allowing scenex to work seamlessly
across desktop and web environments.

The app abstraction decouples scenex from specific GUI frameworks, making it possible
to switch between Qt widgets, WxPython windows, or Jupyter notebook outputs without
changing your visualization code.

Main Components
---------------
- App: Abstract base class for GUI applications
- GuiFrontend: Enumeration of supported GUI frameworks (Qt, WxPython, Jupyter)
- app(): Factory function that returns the active application instance
- determine_app(): Auto-detect which GUI framework to use

Supported Frontends
-------------------
**Qt** (PyQt6, PySide6)
**WxPython**
**Jupyter**

Usage
-----
The app is typically created automatically by `scenex.show()` and/or `scenex.run()`::

    >>> import scenex as snx
    >>> import numpy as np

    >>> # Create a scenex scene
    >>> my_array = np.random.rand(100, 100).astype(np.float32)
    >>> my_scene = snx.Scene(children=[snx.Image(data=my_array)])

    >>> # Showing the scene creates the app if needed
    >>> snx.show(my_scene)
    Canvas(...)
    >>> snx.run()  # Starts the event loop

But it CAN be useful to access the app instance directly. For example, it can be useful
to ask the app to process any pending events::

    >>> from scenex.app import app

    >>> app().process_events()

Notes
-----
This module is designed to be cleanly extractable to a separate library if needed.
It fully encapsulates GUI framework logic and event loop management.

See Also
--------
scenex.run : Convenience function to start the event loop
scenex.show : Creates and displays visualizations
"""

from ._auto import (
    App,
    CursorType,
    GuiFrontend,
    app,
    determine_app,
    ensure_main_thread,
)

# Note that this package is designed to fully encapsulate app logic.
# It is designed to be cleanly extractable to a separate library if needed.

__all__ = [
    "App",
    "CursorType",
    "GuiFrontend",
    "app",
    "determine_app",
    "ensure_main_thread",
]
