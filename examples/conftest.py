"""Pytest setup for notebook tests."""

from typing import TYPE_CHECKING, Any

from nbval.plugin import IPyNbCell  # type: ignore

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# Patch compare_outputs to collapse duplicate RFBOutputContext() outputs.
# Newer versions of vispy may emit more than one RFBOutputContext per display
# call; normalise both sides to at most one.
# ---------------------------------------------------------------------------
_orig_compare_outputs = IPyNbCell.compare_outputs


def _drop_extra_rfb(outputs: list[Any]) -> list[Any]:
    seen = False
    result = []
    for out in outputs:
        if out.get("data", {}).get("text/plain", "") == "RFBOutputContext()":
            if not seen:
                seen = True
                result.append(out)
        else:
            result.append(out)
    return result


def _compare_outputs_filtered(
    self: IPyNbCell,
    test: list[Any],
    ref: list[Any],
    skip_compare: Any = None,
) -> bool:
    # Call the original compare_outputs, with filtered outputs
    return _orig_compare_outputs(  # type: ignore[return-value]
        self, _drop_extra_rfb(test), _drop_extra_rfb(ref), skip_compare=skip_compare
    )


IPyNbCell.compare_outputs = _compare_outputs_filtered  # type: ignore[method-assign]


def pytest_collection_modifyitems(
    session: "pytest.Session", config: "pytest.Config", items: "list[pytest.Item]"
) -> None:
    """Add sanitizers to Notebook tests."""
    for item in items:
        # For each notebook cell being tested...
        if not isinstance(item, IPyNbCell):
            continue
        try:
            # FIXME: On pygfx some lines are printed to stderr that we don't want to
            # compare. Ideally we could ignore ONLY these lines, e.g.
            # libEGL warning: DRI3 error: Could not get DRI3 device
            # libEGL warning: Ensure your X server supports DRI3 to get accelerated...
            # Unable to find extension: VK_EXT_physical_device_drm
            import pygfx  # noqa: F401

            item.parent.skip_compare += ("stderr",)  # pyright: ignore
        except ImportError:
            pass

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
                # -- Collapse multiple RFBOutputContext() lines into one -- #
                r"(RFBOutputContext\(\)\n)+": "RFBOutputContext()\n",
            }
        )
