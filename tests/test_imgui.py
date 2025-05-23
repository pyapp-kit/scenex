from __future__ import annotations

import pytest

import scenex as snx

try:
    from scenex.imgui import add_imgui_controls
except ImportError:
    pytest.skip("imgui_bundle is not installed", allow_module_level=True)


def test_imgui_controls(basic_view: snx.View) -> None:
    """Test that the imgui controls work."""
    snx.show(basic_view)
    add_imgui_controls(basic_view)
    basic_view.render()
