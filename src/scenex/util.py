from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol, no_type_check

from .model import Node, View

if TYPE_CHECKING:

    class SupportsChildren(Protocol):
        """Protocol for node-like objects that have children."""

        @property
        def children(self) -> Iterable[SupportsChildren]:
            """Return the children of the node."""
            ...


__all__ = ["show", "tree_repr"]


# FIXME: this is a temporary solution to deal with pyright and pydantic validation
# not yet sure what the equivalent of pydantic.mypy is for pyright
@no_type_check
def show(obj: Node | View) -> None:
    """Show the scene."""
    from scenex import model

    from .adaptors.auto import get_adaptor_registry

    adaptors = get_adaptor_registry()
    if isinstance(obj, View):
        view = obj
    elif isinstance(obj, model.Scene):
        view = View(scene=obj)
    elif isinstance(obj, Node):
        scene = model.Scene(children=[obj])
        view = View(scene=scene)

    canvas = model.Canvas(views=[view])

    adaptors.get_adaptor(canvas)
    canvas.show()
    cam = adaptors.get_adaptor(view.camera)
    cam._snx_set_range(0.1)


def tree_repr(
    node: SupportsChildren,
    *,
    node_repr: Callable[[Any], str] = object.__repr__,
    prefix: str = "",
    is_last: bool = True,
) -> str:
    """
    Return an ASCII/Unicode tree representation of self and its descendants.

    Assumes `node.children` is an iterable of objects that also implement
    `tree_repr(prefix, is_last)` (or simply this same function via mixin/monkey-patch).
    """
    branch = "└── " if is_last else "├── "
    lines: list[str] = [f"{prefix}{branch}{node_repr(node)}"]
    prefix_child = prefix + ("    " if is_last else "│   ")

    if (children := getattr(node, "children", None)) is None:
        return "\n".join(lines)

    children = _ensure_iterable(children)
    for child, is_last_child in _iter_with_last_flag(children):
        lines.append(
            tree_repr(
                child, node_repr=node_repr, prefix=prefix_child, is_last=is_last_child
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
