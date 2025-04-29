from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol

from scenex import model

if TYPE_CHECKING:

    class SupportsChildren(Protocol):
        """Protocol for node-like objects that have children."""

        @property
        def children(self) -> Iterable[SupportsChildren]:
            """Return the children of the node."""
            ...


__all__ = ["show", "tree_repr"]

logger = logging.getLogger("scenex")


def show(obj: model.Node | model.View | model.Canvas) -> None:
    """Show a scene or view.

    Parameters
    ----------
    obj : Node | View
        The scene or view to show. If a Node is provided, it will be wrapped in a Scene
        and then in a View.
    """
    from .adaptors.auto import get_adaptor_registry

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

        canvas = model.Canvas(views=[view])  # pyright: ignore[reportArgumentType]

    canvas.visible = True
    reg = get_adaptor_registry()
    reg.get_adaptor(canvas, create=True)
    for view in canvas.views:
        cam = reg.get_adaptor(view.camera)
        cam._snx_set_range(0.1)

        native_scene = view.scene._get_native()
        logger.debug("SHOW %s", tree_repr(native_scene))


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
    prefix : str, optional
        Prefix to use for each line of the tree. Defaults to an empty string.
    is_last : bool, optional
        Whether this node is the last child of its parent. Defaults to `True`.
        This is used to determine the branch character to use in the tree
        representation.
    """
    if _prefix:
        branch = "└── " if _is_last else "├── "
    else:
        branch = ""
    lines: list[str] = [f"{_prefix}{branch}{node_repr(node)}"]
    prefix_child = _prefix + ("    " if _is_last else "│   ")

    if (children := getattr(node, "children", None)) is None:
        return "\n".join(lines)

    children = _ensure_iterable(children)
    for child, is_last_child in _iter_with_last_flag(children):
        lines.append(
            tree_repr(
                child,
                node_repr=node_repr,
                _prefix=prefix_child,
                _is_last=is_last_child,
            )
        )
    return "\n".join(lines)


def _iter_with_last_flag(iterable: Iterable) -> Iterator[tuple[Any, bool]]:
    """Yield (item, is_last) for each element in iterable."""
    it = iter(iterable)
    try:
        prev = next(it)
    except StopIteration:
        return
    for curr in it:
        yield prev, False
        prev = curr
    yield prev, True


def _ensure_iterable(obj: object) -> Iterable:
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
