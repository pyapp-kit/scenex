from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

if TYPE_CHECKING:
    from scenex import Node


# Note that scenex follows the inheritance pattern for event subtypes.
# This enables both extensibility, such that new event types can be added easily
# even downstream, and also structured type checking.
@dataclass
class Event:
    """A general interaction event."""

    pass


class MouseButton(IntFlag):
    """A general mouse interaction event."""

    NONE = 0
    LEFT = auto()
    MIDDLE = auto()
    RIGHT = auto()


Intersection: TypeAlias = tuple["Node", float]


class Ray(NamedTuple):
    """A ray passing through the world."""

    origin: tuple[float, float, float]
    direction: tuple[float, float, float]

    def point_at_distance(self, distance: float) -> tuple[float, float, float]:
        x = self.origin[0] + self.direction[0] * distance
        y = self.origin[1] + self.direction[1] * distance
        z = self.origin[2] + self.direction[2] * distance
        return (x, y, z)

    def intersections(self, graph: Node) -> list[Intersection]:
        """
        Find all intersections of this ray with the given scene graph.

        Returns a list of (node, distance) tuples, sorted by distance.
        """
        through: list[Intersection] = []
        for child in graph.children:
            if (d := child.passes_through(self)) is not None:
                through.append((child, d))
                through.extend(self.intersections(child))
        return sorted(through, key=lambda inter: inter[1])


@dataclass
class ResizeEvent(Event):
    """A window resize event."""

    width: int  # in pixels
    height: int  # in pixels


@dataclass
class MouseEvent(Event):
    """Base class for mouse interaction events."""

    canvas_pos: tuple[float, float]
    world_ray: Ray
    buttons: MouseButton


@dataclass
class MouseLeaveEvent(Event):
    """Mouse leave event.

    Note that this does not inherit from MouseEvent, as no position or buttons are
    """

    pass


@dataclass
class MouseEnterEvent(MouseEvent):
    """Mouse enter event."""

    pass


@dataclass
class MouseMoveEvent(MouseEvent):
    """Mouse move event."""

    pass


@dataclass
class MousePressEvent(MouseEvent):
    """Mouse press event."""

    pass


@dataclass
class MouseReleaseEvent(MouseEvent):
    """Mouse release event."""

    pass


@dataclass
class MouseDoublePressEvent(MouseEvent):
    """Mouse double press event."""

    pass


@dataclass
class WheelEvent(MouseEvent):
    """A mouse interaction event describing wheel movement."""

    angle_delta: tuple[float, float]


class EventFilter:
    def uninstall(self) -> None:
        """Uninstall the event filter."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    pass
