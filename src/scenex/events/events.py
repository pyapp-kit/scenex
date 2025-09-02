from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto
from typing import NamedTuple


@dataclass
class Event:
    """A general interaction event."""

    # TODO: Enum?
    type: str

    def __eq__(self, value):
        if not isinstance(value, Event):
            return NotImplemented
        return self.type == value.type


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
