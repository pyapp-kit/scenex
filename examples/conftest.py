"""Pytest setup for notebook tests."""

from typing import TYPE_CHECKING

from nbval.plugin import IPyNbCell  # type: ignore

if TYPE_CHECKING:
    import pytest


def pytest_collection_modifyitems(
    session: "pytest.Session", config: "pytest.Config", items: "list[pytest.Item]"
) -> None:
    """Add sanitizers to Notebook tests."""
    for item in items:
        # For each notebook cell being tested...
        if not isinstance(item, IPyNbCell):
            continue

        item.parent.skip_compare += ("stderr",)  # pyright: ignore
        # Add the following (regex) sanitations...
        # (Note that both the expected and actual cell outputs will be sanitized)
        item.parent.sanitize_patterns.update(  # pyright: ignore
            {
                # -- Convert all jupyter canvases to the same (pygfx) class name -- #
                r"CanvasBackend": "JupyterRenderCanvas",
                # -- Convert canvas sizes to integers (vispy uses floats) -- #
                r"(\d+)\.0px": r"\1px",  # e.g. 600.0px -> 600px
                # -- Normalize div snapshot ids (unique to each run) -- #
                r"snapshot-[a-f0-9]+": "snapshot-XXXXX",
                # -- Normalize canvas data (within the img tag) -- #
                r"data:image/png;base64,[A-Za-z0-9+/=]+": "data:image/png;base64,XXXXX",
            }
        )
