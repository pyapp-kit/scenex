from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto
from typing import NamedTuple


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


class Ray(NamedTuple):
    """A ray passing through the world."""

    origin: tuple[float, float, float]
    direction: tuple[float, float, float]

    def point_at_distance(self, distance: float) -> tuple[float, float, float]:
        x = self.origin[0] + self.direction[0] * distance
        y = self.origin[1] + self.direction[1] * distance
        z = self.origin[2] + self.direction[2] * distance
        return (x, y, z)


@dataclass
class MouseEvent(Event):
    """Base class for mouse interaction events."""

    canvas_pos: tuple[float, float]
    world_ray: Ray
    buttons: MouseButton


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
