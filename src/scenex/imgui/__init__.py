"""ImGui controls for SceneX."""

from typing import Any

from ._controls import add_imgui_controls

__all__ = ["add_imgui_controls"]


def __getattr__(name: str) -> Any:
    """Lazy load the imgui module."""
    if name == "add_imgui_controls":
        from ._controls import add_imgui_controls

        return add_imgui_controls
    raise AttributeError(f"module {__name__} has no attribute {name}")
