from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

if TYPE_CHECKING:
    from scenex import Node, View


@dataclass
class Event:
    """Base class for all user interaction and system events.

    Event is the root of the event hierarchy in scenex. All specific event types
    (mouse, keyboard, resize) inherit from this base class, enabling polymorphic
    event handling and extensibility for custom event types.

    The inheritance-based design allows:
    - Type checking with isinstance() to discriminate event types
    - Extensibility for adding new event types downstream
    - Structured event filtering based on event class hierarchy

    See Also
    --------
    MouseEvent : Base class for mouse-related events
    ResizeEvent : Window resize event
    """

    pass


class MouseButton(IntFlag):
    """Enumeration of mouse button states as bit flags.

    MouseButton uses IntFlag to allow bitwise operations, enabling representation
    of multiple simultaneous button presses.

    Examples
    --------
    Check if left button is pressed:
        >>> event = MousePressEvent(
        ...     canvas_pos=(100, 150),
        ...     world_ray=Ray(origin=(0, 0, 0), direction=(0, 0, -1), source=None),
        ...     buttons=MouseButton.LEFT | MouseButton.RIGHT,
        ... )
        >>> if event.buttons & MouseButton.LEFT:
        ...     print("Left button is down")
        Left button is down

    Check for specific button combination:
        >>> if event.buttons == (MouseButton.LEFT | MouseButton.RIGHT):
        ...     print("Both left and right buttons pressed")
        Both left and right buttons pressed

    Check if any button is pressed:
        >>> if event.buttons != MouseButton.NONE:
        ...     print("Some button is pressed")
        Some button is pressed
    """

    NONE = 0
    LEFT = auto()
    MIDDLE = auto()
    RIGHT = auto()


Intersection: TypeAlias = tuple["Node", float]


class Ray(NamedTuple):
    """A 3D ray in world space representing a mouse position.

    A Ray represents the path of the mouse cursor projected into 3D world space,
    starting from the camera and passing through the cursor position on the view.
    Rays are the fundamental mechanism for 3D picking and intersection testing,
    allowing determination of which scene objects are under the mouse cursor.

    The ray is defined by an origin point (typically the camera position) and a
    normalized direction vector. All MouseEvent instances include a world_ray that
    can be used to test intersections with scene geometry.

    Attributes
    ----------
    origin : tuple[float, float, float]
        The starting point of the ray in world coordinates, typically the camera
        position for perspective projections or a point on the view plane for
        orthographic projections.
    direction : tuple[float, float, float]
        The normalized direction vector of the ray in world coordinates. For
        perspective views, this points from the camera through the cursor. For
        orthographic views, this is parallel to the camera's view direction.
    source : View
        The view that generated this ray, providing context for which camera
        and scene the ray originated from.

    Examples
    --------
    Find all intersections with a scene:
        >>> import numpy as np
        >>> import scenex as snx
        >>> view = snx.View(
        ...     scene=snx.Scene(
        ...         children=[
        ...             snx.Image(data=np.random.rand(100, 100)),
        ...             snx.Points(
        ...                 vertices=np.asarray([[0, 0, 0], [1, 1, 0]]),
        ...                 size=5,
        ...                 edge_width=0,
        ...             ),
        ...         ]
        ...     )
        ... )
        >>> ray = Ray(origin=(1, 1, 10), direction=(0, 0, -1), source=view)
        >>> ray.intersections(view.scene)
        [(Points(...), 7.5), (Image(...), 10.0)]

    See Also
    --------
    MouseEvent : Events that include world_ray
    Node.passes_through : Node method for computing ray intersections
    """

    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    source: View

    def point_at_distance(self, distance: float) -> tuple[float, float, float]:
        """Compute the 3D point at a given distance along the ray.

        Parameters
        ----------
        distance : float
            The distance along the ray from the origin. Positive values extend
            in the direction of the ray, negative values extend backward from
            the origin.

        Returns
        -------
        tuple[float, float, float]
            The (x, y, z) coordinates of the point at the specified distance
            along the ray.
        """
        x = self.origin[0] + self.direction[0] * distance
        y = self.origin[1] + self.direction[1] * distance
        z = self.origin[2] + self.direction[2] * distance
        return (x, y, z)

    def intersections(self, graph: Node) -> list[Intersection]:
        """Find all nodes intersected by this ray in the scene graph.

        Recursively tests the ray against the given node and all its descendants,
        returning all intersections sorted by distance from the ray origin. Only
        visible nodes are tested.

        Parameters
        ----------
        graph : Node
            The root node to test. Typically a Scene, but can be any node with
            children.

        Returns
        -------
        list[Intersection]
            List of (node, distance) tuples for all intersections, sorted by
            increasing distance from the ray origin. The distance is the parameter
            t where intersection occurs at origin + t * direction.
        """
        through: list[Intersection] = []
        if graph.visible:
            # ...check the node itself...
            if (d := graph.passes_through(self)) is not None:
                through.append((graph, d))
            # ...then check its children...
            for child in graph.children:
                through.extend(self.intersections(child))
        return sorted(through, key=lambda inter: inter[1])


@dataclass
class ResizeEvent(Event):
    """Canvas window resize event.

    Fired when the canvas window changes dimensions, whether from user interaction
    (dragging window edges), programmatic resizing, or window manager actions. This
    event allows views and other components to adapt to new canvas dimensions.

    Attributes
    ----------
    width : int
        The new width of the canvas in pixels.
    height : int
        The new height of the canvas in pixels.
    """

    width: int  # in pixels
    height: int  # in pixels


@dataclass
class MouseEvent(Event):
    """Base class for all mouse-related interaction events.

    MouseEvent provides common fields for all mouse interactions, including the
    2D canvas position, the 3D world ray for picking, and the state of mouse buttons.
    Specific mouse event types (move, press, release, etc.) inherit from this base.

    Attributes
    ----------
    canvas_pos : tuple[float, float]
        The (x, y) position of the mouse cursor in canvas pixel coordinates, with
        origin at the top-left corner.
    world_ray : Ray
        The 3D ray in world space corresponding to this mouse position, used for
        3D picking and intersection testing. The ray passes from the camera through
        the cursor position.
    buttons : MouseButton
        Bit flags indicating which mouse buttons are currently pressed. Use bitwise
        operations to test button states (e.g., buttons & MouseButton.LEFT).

    See Also
    --------
    MouseMoveEvent : Mouse cursor movement
    MousePressEvent : Mouse button press
    MouseReleaseEvent : Mouse button release
    WheelEvent : Mouse wheel scroll
    Ray : 3D ray for picking
    """

    canvas_pos: tuple[float, float]
    world_ray: Ray
    buttons: MouseButton


@dataclass
class MouseLeaveEvent(Event):
    """Mouse cursor leaving the view area.

    Fired when the mouse cursor exits the bounds of a view. This is distinct from
    other mouse events in that it does not include position or button information,
    as the cursor is no longer over the view.

    Note that this does not inherit from MouseEvent, as no position or buttons are
    available when the cursor has left the view.

    See Also
    --------
    MouseEnterEvent : Mouse cursor entering the view
    """

    pass


@dataclass
class MouseEnterEvent(MouseEvent):
    """Mouse cursor entering the view area.

    Fired when the mouse cursor enters the bounds of a view from outside. Includes
    the entry position and button states.

    See Also
    --------
    MouseLeaveEvent : Mouse cursor leaving the view
    """

    pass


@dataclass
class MouseMoveEvent(MouseEvent):
    """Mouse cursor movement within the view.

    Fired when the mouse cursor moves within the view bounds. Includes the current
    position, world ray, and button states. This event fires continuously during
    cursor movement.
    """

    pass


@dataclass
class MousePressEvent(MouseEvent):
    """Mouse button press.

    Fired when a mouse button is pressed down. The buttons field indicates which
    button(s) are now pressed. For detecting which button was newly pressed, compare
    with previous button states.

    See Also
    --------
    MouseReleaseEvent : Mouse button release
    MouseDoublePressEvent : Double-click detection
    """

    pass


@dataclass
class MouseReleaseEvent(MouseEvent):
    """Mouse button release.

    Fired when a mouse button is released. The buttons field reflects the state
    after the release (i.e., the released button is no longer set in the flags).

    See Also
    --------
    MousePressEvent : Mouse button press
    """

    pass


@dataclass
class MouseDoublePressEvent(MouseEvent):
    """Mouse button double-click.

    Fired when a mouse button is double-clicked (pressed twice in rapid succession).
    The timing threshold for double-click detection is system-dependent.

    See Also
    --------
    MousePressEvent : Single mouse button press
    """

    pass


@dataclass
class WheelEvent(MouseEvent):
    """Mouse wheel scroll event.

    Fired when the mouse wheel (or trackpad scroll) is used. Includes the scroll
    delta in both horizontal and vertical directions. The magnitude and units of
    angle_delta are platform-dependent but typically represent degrees or steps.

    Attributes
    ----------
    angle_delta : tuple[float, float]
        The (horizontal, vertical) scroll delta. Positive vertical values typically
        represent scrolling up/away from the user, negative values down/toward the
        user. Horizontal scrolling (if supported) uses the first component.
    """

    angle_delta: tuple[float, float]


class EventFilter:
    """Base class for event filter handles.

    EventFilter instances are returned when installing event filters on views or
    canvases. They provide a mechanism to uninstall the filter when it's no longer
    needed, ensuring proper cleanup and preventing memory leaks.
    """

    def uninstall(self) -> None:
        """Remove this event filter.

        Uninstalls the event filter, ensuring that the filter function will no
        longer be called for future events. After calling uninstall(), this
        EventFilter instance should not be used further.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    pass
