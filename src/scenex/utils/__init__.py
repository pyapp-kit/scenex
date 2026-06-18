"""Utility functions for displaying and debugging scenex visualizations.

This module provides helper functions for common visualization tasks including
displaying models, formatting scene graph trees, and utility functions used
internally by scenex.

The `show()` function is the primary entry point for creating visualizations,
handling the details of canvas creation, backend selection, and camera fitting
automatically for a provided node, view or canvas.
"""

from ._app import native, run, set_cursor, show

__all__ = ["native", "run", "set_cursor", "show"]
