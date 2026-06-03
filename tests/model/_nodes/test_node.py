from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import scenex as snx

# Avoid making new vertices every time, save some space.
_VERTICES = np.zeros((1, 3), dtype=np.float32)


def test_add_child() -> None:
    """Adding a child should set the child's parent, add it to the parent's children."""
    parent = snx.Scene()
    child = snx.Points(vertices=_VERTICES)
    event_listener = MagicMock()
    parent.child_added.connect(event_listener)

    parent.add_child(child)

    assert child.parent is parent
    assert child in parent.children
    assert len(parent.children) == 1
    event_listener.assert_called_once_with(child)
    event_listener.reset_mock()

    parent.add_child(child)

    assert parent.children.count(child) == 1
    assert len(parent.children) == 1
    event_listener.assert_not_called()


def test_remove_child() -> None:
    """Removing a child clears its parent and emits the signal.

    Should be a no-op if the child is not present.
    """
    parent = snx.Scene()
    c1, c2 = snx.Points(vertices=_VERTICES), snx.Points(vertices=_VERTICES)
    parent.add_child(c1)
    parent.add_child(c2)
    mock = MagicMock()
    parent.child_removed.connect(mock)

    parent.remove_child(c1)

    assert c1 not in parent.children
    assert c1.parent is None
    assert c2 in parent.children  # sibling unaffected
    mock.assert_called_once_with(c1)
    mock.reset_mock()

    # removing an absent child should be a silent no-op
    parent.remove_child(c1)
    mock.assert_not_called()


def test_constructor_children_kwarg() -> None:
    """children= kwarg should populate children and set each child's parent."""
    c1, c2 = snx.Points(vertices=_VERTICES), snx.Points(vertices=_VERTICES)

    parent = snx.Scene(children=[c1, c2])

    assert parent.children == (c1, c2)
    assert c1.parent is parent
    assert c2.parent is parent


def test_constructor_parent_kwarg() -> None:
    """parent= kwarg should register the node as a child and emit child_added."""
    parent = snx.Scene()
    mock = MagicMock()
    parent.child_added.connect(mock)

    child = snx.Scene(parent=parent)

    assert child in parent.children
    assert child.parent is parent
    mock.assert_called_once_with(child)

    # default: no parent
    assert snx.Scene().parent is None


def test_reparent() -> None:
    """Moving a child from one parent to another should update both parents."""
    p1, p2 = snx.Scene(), snx.Scene()
    child = snx.Points(vertices=_VERTICES)
    p1.add_child(child)

    # reparent via add_child
    p2.add_child(child)

    assert child not in p1.children
    assert child in p2.children
    assert child.parent is p2

    # reparent via parent field
    child.parent = p1

    assert child in p1.children
    assert child not in p2.children
    assert child.parent is p1


def test_children() -> None:
    """children should be an empty tuple by default and reflect add/remove calls."""
    parent = snx.Scene()
    assert parent.children == ()

    child = snx.Points(vertices=_VERTICES)
    parent.add_child(child)
    assert parent.children == (child,)  # type: ignore

    parent.remove_child(child)
    assert parent.children == ()


def test_contains() -> None:
    """__contains__ checks direct children only, not deeper descendants."""
    grandparent, parent = snx.Scene(), snx.Scene()
    child = snx.Points(vertices=_VERTICES)
    other = snx.Points(vertices=_VERTICES)

    grandparent.add_child(parent)
    parent.add_child(child)

    assert parent in grandparent
    assert other not in grandparent
    assert child not in grandparent  # grandchild — not a direct child


def test_node_cannot_be_instantiated_directly() -> None:
    from scenex.model._nodes.node import Node

    with pytest.raises(TypeError, match="Node cannot be instantiated directly"):
        Node()
