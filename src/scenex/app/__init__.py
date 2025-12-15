"""The Scenex App Abstraction."""

from ._auto import App, GuiFrontend, app, determine_app

# Note that this package is designed to fully encapsulate app logic.
# It is designed to be cleanly extractable to a separate library if needed.

__all__ = ["App", "GuiFrontend", "app", "determine_app"]
