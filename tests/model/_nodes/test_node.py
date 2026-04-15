from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

import scenex as snx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_points() -> snx.Points:
    return snx.Points(vertices=np.zeros((1, 3), dtype=np.float32))


def make_scene() -> snx.Scene:
    return snx.Scene()


def test_add_child() -> None:
    """Adding a child should set the child's parent, add it to the parent's children."""
    parent = make_scene()
    child = make_points()
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
    parent = make_scene()
    c1, c2 = make_points(), make_points()
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
    c1, c2 = make_points(), make_points()

    parent = snx.Scene(children=[c1, c2])

    assert parent.children == (c1, c2)
    assert c1.parent is parent
    assert c2.parent is parent


def test_constructor_parent_kwarg() -> None:
    """parent= kwarg should register the node as a child and emit child_added."""
    parent = make_scene()
    mock = MagicMock()
    parent.child_added.connect(mock)

    child = snx.Scene(parent=parent)

    assert child in parent.children
    assert child.parent is parent
    mock.assert_called_once_with(child)

    # default: no parent
    assert make_scene().parent is None


def test_reparent() -> None:
    """Moving a child from one parent to another should update both parents."""
    p1 = make_scene()
    p2 = make_scene()
    child = make_points()
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
    parent = make_scene()
    assert parent.children == ()

    child = make_points()
    parent.add_child(child)
    assert parent.children == (child,)  # type: ignore

    parent.remove_child(child)
    assert parent.children == ()


def test_contains() -> None:
    """__contains__ checks direct children only, not deeper descendants."""
    grandparent = make_scene()
    parent = make_scene()
    child = make_points()
    other = make_points()

    grandparent.add_child(parent)
    parent.add_child(child)

    assert parent in grandparent
    assert other not in grandparent
    assert child not in grandparent  # grandchild — not a direct child
