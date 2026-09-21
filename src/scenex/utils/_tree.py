"""Utility functions for displaying and debugging scenex scenegraphs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from typing import TypeAlias

    Tree: TypeAlias = str | dict[str, list["Tree"]]

    class SupportsChildren(Protocol):
        """Protocol for node-like objects that have children."""

        @property
        def children(self) -> Iterable[SupportsChildren]:
            """Return the children of the node."""
            ...


def _cls_name_with_id(obj: Any) -> str:
    return f"{obj.__class__.__name__}:{id(obj)}"


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


def _ensure_iterable(obj: object) -> Iterable[Any]:
    if isinstance(obj, Iterable):
        return obj
    if callable(obj):
        with suppress(TypeError):
            return _ensure_iterable(obj())
    raise TypeError(
        f"Expected an iterable or callable that returns an iterable, "
        f"got {type(obj).__name__}"
    )
