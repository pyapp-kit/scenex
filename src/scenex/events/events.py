from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pylinalg as la

# from scenex.model import Camera

if TYPE_CHECKING:
    from typing import Any

    from scenex.model import Canvas, Node, View


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
        if view := _containing_view(event.canvas_pos, canvas):
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

    return handled


def _containing_view(pos: tuple[float, float], canvas: Canvas) -> View | None:
    for view in canvas.views:
        if pos in view.layout:
            return view
    return None


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


def _canvas_to_world(canvas: Canvas, canvas_pos: tuple[float, float]) -> Ray | None:
    """Map XY canvas position (pixels) to XYZ coordinate in world space."""
    # Code adapted from:
    # https://github.com/pygfx/pygfx/pull/753/files#diff-173d643434d575e67f8c0a5bf2d7ea9791e6e03a4e7a64aa5fa2cf4172af05cdR395
    view = _containing_view(canvas_pos, canvas)
    if view is None:
        return None

    # Get position relative to viewport
    pos_rel = (
        canvas_pos[0] - view.layout.x,
        canvas_pos[1] - view.layout.y,
    )

    width, height = view.layout.size

    # Convert position to Normalized Device Coordinates (NDC) - i.e., within [-1, 1]
    x = pos_rel[0] / width * 2 - 1
    y = -(pos_rel[1] / height * 2 - 1)
    pos_ndc = (x, y)

    # Note that the camera matrix is the matrix multiplication of:
    # * The projection matrix, which projects local space (the rectangular
    #   bounds of the perspective camera) into NDC.
    # * The view matrix, i.e. the transform positioning the camera in the world.
    # The result is a matrix mapping world coordinates
    camera_matrix = view.camera.projection @ view.camera.transform.inv().T
    pos_diff = la.vec_transform(view.camera.transform.root[3, :3], camera_matrix.T)
    # Unproject the canvas NDC coordinates into world space.
    pos_world = la.vec_unproject(pos_ndc + pos_diff[:2], camera_matrix)

    # To find the direction of the ray, we find a unprojected point farther away
    # and subtract the closer point.
    pos_world_farther = la.vec_unproject(pos_ndc + pos_diff[:2], camera_matrix, depth=1)
    direction = pos_world_farther - pos_world
    direction = direction / np.linalg.norm(direction)

    ray = Ray(
        origin=tuple(pos_world),
        direction=tuple(direction),
    )
    return ray
