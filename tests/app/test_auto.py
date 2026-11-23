"""Tests for excepthook functionality in scenex.app._auto."""

import pytest

from scenex.app import GuiFrontend, app, determine_app


@pytest.mark.skipif(
    determine_app() == GuiFrontend.JUPYTER,
    reason="IPython has its own exception handling.",
)
def test_excepthook(capsys: pytest.CaptureFixture) -> None:
    """Tests that exceptions raised in the app's event loop don't crash the app."""
    qapp = app()
    # NOTE this create_app call is needed to counter pytestqt's patch of sys.excepthook
    qapp.create_app()

    def raise_exception() -> None:
        raise NotImplementedError("Test exception")

    qapp.call_later(0, raise_exception)
    qapp.process_events()

    captured = capsys.readouterr()
    assert "NotImplementedError" in captured.err
    assert "Test exception" in captured.err
