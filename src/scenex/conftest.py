"""Pytest setup for doctests."""

from collections.abc import Iterator
from contextlib import ExitStack
from unittest.mock import Mock, patch

import pytest

import scenex as snx


@pytest.fixture(autouse=True)
def _doctest_setup(doctest_namespace: dict) -> Iterator[None]:
    """Sets up the doctest namespace.

    The main function currently is to allow examples to call blocking functions (for
    streamlined copy-paste) without actually blocking.
    """
    # Mock snx.run
    snx.run = Mock(return_value=None)

    with ExitStack() as stack:
        try:
            # Mock IPython display function IFF we're testing Jupyter
            # Need to patch where it's used, not where it's defined
            from jupyter_rfb import widget

            stack.enter_context(
                patch.object(widget, "display", Mock(return_value=None))
            )
            from IPython import display

            stack.enter_context(
                patch.object(display, "display", Mock(return_value=None))
            )
        except ImportError:
            pass

        # TODO: Necessary for the work on https://github.com/pyapp-kit/scenex/pull/42
        # # Mock app().run
        # original_app = app()

        # # Create a wrapper that delegates everything to the real app except run()
        # class MockedApp:
        #     def run(self) -> None:
        #         """No-op run method for doctests."""
        #         pass

        #     def run(self) -> None:
        #         """No-op run method for doctests."""
        #         pass

        #     def __getattr__(self, name: str) -> Any:
        #         """Delegate all other attributes to the real app."""
        #         return getattr(original_app, name)

        # doctest_namespace["app"] = MockedApp

        yield
