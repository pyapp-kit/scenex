from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Any

    from scenex.model import Canvas, Node


@dataclass
class Event:
    """A general interaction event."""

    # TODO: Enum?
    type: str


class MouseButton(IntFlag):
    """A general mouse interaction event."""

    NONE = 0
    LEFT = auto()
    MIDDLE = auto()
    RIGHT = auto()


class Ray(NamedTuple):
    """A ray passing through the world."""

    origin: tuple[float, float, float]
    direction: tuple[float, float, float]


@dataclass
class MouseEvent(Event):
    """A general mouse interaction event."""

    type: str
    canvas_pos: tuple[float, float]
    world_ray: Ray
    # TODO: Enum?
    # TODO: Just a MouseButton, you can AND the MouseButtons
    buttons: MouseButton


@dataclass
class WheelEvent(MouseEvent):
    """A mouse interaction event describing wheel movement."""

    angle_delta: tuple[float, float]


def _handle_event(canvas: Canvas, event: Event) -> bool:
    handled = False
    if isinstance(event, MouseEvent):
        if view := canvas._containing_view(event.canvas_pos):
            through: list[tuple[Node, float]] = []
            for child in view.scene.children:
                if (d := child.passes_through(event.world_ray)) is not None:
                    through.append((child, d))

            # FIXME: Consider only reporting the first?
            # Or do we only report until we hit a node with opacity=1?
            for node, _depth in sorted(through, key=lambda e: e[1]):
                # Filter through parent scenes to child
                handled |= _filter_through(event, node, node)
            # No nodes in the view handled the event - pass it to the camera
            if not handled and view.camera.interactive:
                handled |= view.camera.filter_event(event, view.camera)

    canvas._get_adaptors()
    return handled


def _filter_through(event: Any, node: Node, target: Node) -> bool:
    """Filter the event through the scene graph to the target node."""
    # TODO: Suppose a scene is not interactive. If the node is interactive, should it
    # receive the event?

    # First give this node a chance to filter the event.

    if node.interactive and node.filter_event(event, target):
        # Node filtered out the event, so we stop here.
        return True
    if (parent := node.parent) is None:
        # Node did not filter out the event, and we've reached the top of the graph.
        return False
    # Recursively filter the event through node's parent.
    return _filter_through(event, parent, target)
