"""Utility functions for displaying and debugging scenex visualizations.

This module provides helper functions for common visualization tasks including
displaying models, formatting scene graph trees, and utility functions used
internally by scenex.

The `show()` function is the primary entry point for creating visualizations,
handling the details of canvas creation, backend selection, and camera fitting
automatically for a provided node, view or canvas.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol

from scenex import model
from scenex.app import app
from scenex.utils import projections

if TYPE_CHECKING:
    from typing import TypeAlias

    Tree: TypeAlias = str | dict[str, list["Tree"]]

    class SupportsChildren(Protocol):
        """Protocol for node-like objects that have children."""

        @property
        def children(self) -> Iterable[SupportsChildren]:
            """Return the children of the node."""
            ...


__all__ = ["show", "tree_dict", "tree_repr"]

logger = logging.getLogger("scenex")


def tree_repr(
    node: SupportsChildren,
    *,
    node_repr: Callable[[Any], str] = object.__repr__,
    _prefix: str = "",
    _is_last: bool = True,
) -> str:
    """
    Return an ASCII/Unicode tree representation of `node` and its descendants.

    This assumes that `node` is a tree-like object with a `children` attribute that is
    either a property or a callable that returns an iterable of child nodes.

    Parameters
    ----------
    node : SupportsChildren
        Any object that has a `children` attribute or method that returns an iterable
        of child nodes.
    node_repr : Callable[[Any], str], optional
        Function to convert the node to a string. Defaults to `object.__repr__` (which
        avoids complex repr functions on objects, but use `repr` if you want to see
        the full representation).
    _prefix : str, optional
        Prefix to use for each line of the tree. Defaults to an empty string.
    _is_last : bool, optional
        Whether this node is the last child of its parent. Defaults to `True`.
        This is used to determine the branch character to use in the tree
        representation.
    """
    if _prefix:
        branch = "└── " if _is_last else "├── "
    else:
        branch = ""

    lines: list[str] = [f"{_prefix}{branch}{node_repr(node)}"]
    if children := list(_get_children(node)):
        prefix_child = _prefix + ("    " if _is_last else "│   ")
        for idx, child in enumerate(children):
            lines.append(
                tree_repr(
                    child,
                    node_repr=node_repr,
                    _prefix=prefix_child,
                    _is_last=idx == len(children) - 1,
                )
            )
    return "\n".join(lines)


def _ensure_iterable(obj: object) -> Iterable[Any]:
    """Ensure the object is iterable."""
    if isinstance(obj, Iterable):
        return obj
    if callable(obj):
        with suppress(TypeError):
            return _ensure_iterable(obj())
    raise TypeError(
        f"Expected an iterable or callable that returns an iterable, "
        f"got {type(obj).__name__}"
    )


def show(
    obj: model.Node | model.View | model.Canvas, *, backend: str | None = None
) -> model.Canvas:
    """Display a visualization by creating a canvas and making it visible.

    This is the primary function for creating and displaying scenex visualizations.
    It accepts nodes, views, or canvases, automatically wrapping them in the necessary
    container objects and creating the appropriate backend adaptors.

    The function automatically fits the camera view to show all visible content and
    makes the canvas window visible. After calling `show()`, use `run()` to enter
    the event loop (in desktop applications) or continue working (in notebooks).

    Parameters
    ----------
    obj : Node | View | Canvas
        The object to visualize:
        - Node (Image, Points, Line, etc.): Wrapped in Scene and View automatically
        - Scene: Wrapped in a View with a default Camera
        - View: Placed on a new Canvas
        - Canvas: Displayed directly (already contains Views)
    backend : str | None, optional
        Graphics backend to use ("pygfx" or "vispy"). If None, uses the backend
        specified by `use()`, `SCENEX_CANVAS_BACKEND` environment variable, or
        auto-detection. Default is `None`.

    Returns
    -------
    Canvas
        The canvas containing the visualization. Can be used to further manipulate
        the display or access the created views.

    Examples
    --------
    Show a simple image:
        >>> import numpy as np
        >>> import scenex as snx
        >>> data = np.random.rand(100, 100)
        >>> img = snx.Image(data=data)
        >>> snx.show(img)
        >>> snx.run()

    Show a scene with multiple objects:
        >>> scene = snx.Scene(
        ...     children=[
        ...         snx.Image(data=image_data),
        ...         snx.Points(coords=points, face_color=Color("red")),
        ...     ]
        ... )
        >>> canvas = snx.show(scene)
        >>> snx.run()

    Show a view with interactive camera:
        >>> view = snx.View(
        ...     scene=my_scene,
        ...     camera=snx.Camera(controller=snx.PanZoom(), interactive=True),
        ... )
        >>> snx.show(view)
        >>> snx.run()

    Show with specific backend:
        >>> snx.show(my_scene, backend="pygfx")
        >>> snx.run()

    Access the returned canvas:
        >>> canvas = snx.show(my_image)
        >>> canvas.background_color = Color("white")
        >>> canvas.width = 800
        >>> snx.run()

    Notes
    -----
    - The camera is automatically zoomed to fit all visible content with 90% coverage
    - Canvas size defaults to the view's layout dimensions
    - Call `run()` after `show()` to enter the event loop in desktop applications
    - In Jupyter notebooks, visualizations appear automatically without `run()`
    """
    from .adaptors import get_adaptor_registry

    view = None
    if isinstance(obj, model.Canvas):
        canvas = obj
    else:
        if isinstance(obj, model.View):
            view = obj
        elif isinstance(obj, model.Scene):
            view = model.View(scene=obj)
        elif isinstance(obj, model.Node):
            scene = model.Scene(children=[obj])
            view = model.View(scene=scene)

        canvas = model.Canvas(
            # Respect the view size if provided
            width=int(view.layout.width),
            height=int(view.layout.height),
        )
        canvas.grid.add(view, row=0, col=0)

    canvas.visible = True
    reg = get_adaptor_registry(backend=backend)
    reg.get_adaptor(canvas, create=True)
    app().create_app()
    for view in canvas.views:
        projections.zoom_to_fit(view, zoom_factor=0.9)

        # logger.debug("SHOW MODEL  %s", tree_repr(view.scene))
        # native_scene = view.scene._get_native()
        # logger.debug("SHOW NATIVE %s", tree_repr(native_scene))
    return canvas


def _cls_name_with_id(obj: Any) -> str:
    return f"{obj.__class__.__name__}:{id(obj)}"


def tree_dict(
    node: SupportsChildren,
    *,
    obj_name: Callable[[Any], str] = _cls_name_with_id,
) -> Tree:
    """Build an intermediate representation of the tree rooted at `node`.

    Leaves are represented as strings, and non-leaf nodes are represented as
    dictionaries with the node name as the key and a list of child nodes as the value.
    This is useful for debugging and visualization purposes.

    Parameters
    ----------
    node : SupportsChildren
        The root node of the tree to be represented.
    obj_name : Callable[[Any], str], optional
        A function to convert the node to a string. Defaults to a lambda function that
        returns the class name and ID

    Returns
    -------
    str | dict[str, list[dict | str]]
        A string, if the node is a leaf, or a dictionary representing the tree,
        if the node has children, like `{"node_name": ["child1", "child2", ...]}`.
    """
    node_name = obj_name(node)
    if not (children := _get_children(node)):
        return node_name

    result: list[dict | str] = []
    for child in children:
        result.append(tree_dict(child, obj_name=obj_name))
    return {obj_name(node): result}


def _get_children(obj: Any) -> Iterable[Any]:
    if (children := getattr(obj, "children", None)) is None:
        return ()
    return _ensure_iterable(children)
