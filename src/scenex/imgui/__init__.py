"""ImGui controls for interactive scenex visualization.

This module provides ImGui-based interactive controls for scenex scenes. It adds an
overlay widget panel that allows real-time manipulation of scene parameters, view
properties, and node attributes through sliders, checkboxes, color pickers, and other
widgets.

The controls are automatically generated from Pydantic model fields, providing a
consistent interface without requiring manual widget creation.

Requirements
------------
This module requires additional dependencies::

    pip install scenex[imgui]

This installs imgui_bundle and pygfx with ImGui support.

Main Function
-------------
add_imgui_controls : function
    Add an interactive ImGui control panel to a view

Example
-------
Add controls to a scene with an image::

    import scenex as snx
    from scenex.imgui import add_imgui_controls

    # Create a scene with some content
    image = snx.Image(data=my_array)
    view = snx.View(scene=snx.Scene(children=[image]))

    # Add interactive controls
    add_imgui_controls(view)

    # Show and run
    snx.show(view)
    snx.run()

The control panel will display collapsible sections for the view and each child node,
with automatically generated widgets for adjusting properties like opacity, colors,
transforms, and node-specific parameters.

Notes
-----
Only works with pygfx backend
"""

from typing import Any

from ._controls import add_imgui_controls

__all__ = ["add_imgui_controls"]


def __getattr__(name: str) -> Any:
    """Lazy load the imgui module."""
    if name == "add_imgui_controls":
        from ._controls import add_imgui_controls

        return add_imgui_controls
    raise AttributeError(f"module {__name__} has no attribute {name}")
