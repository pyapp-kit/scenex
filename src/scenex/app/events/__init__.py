"""Event system for handling user input and interaction.

This module provides a unified event abstraction for user interactions across different
GUI frameworks and rendering backends. Events represent user actions (e.g. mouse clicks,
movement, scrolling) and system events (e.g. window resize), allowing nodes and
controllers to respond to input in a framework-agnostic way.

Event Types
-----------
**Mouse Events**:
    - MousePressEvent: Mouse button pressed
    - MouseReleaseEvent: Mouse button released
    - MouseMoveEvent: Mouse cursor moved
    - MouseDoublePressEvent: Mouse button double-pressed
    - MouseEnterEvent: Mouse entered a view
    - MouseLeaveEvent: Mouse left a view
    - WheelEvent: Mouse wheel scrolled

**System Events**:
    - ResizeEvent: Canvas window resized

Event Flow
----------
Events are dispatched by the canvas to views and their camera controllers::

    Canvas → View (filter_event) → Camera Controller (handle_event)

The flow works as follows:
1. Canvas determines which view contains the cursor position
2. Canvas calls the view's filter_event() method with the event
3. If the view's camera is interactive, the camera controller's handle_event()
   is called
4. Handlers return True to stop propagation or False to continue

Key Concepts
------------
**Ray**: 3D ray in world space representing the mouse position
    - Used for 3D picking and intersection tests
    - Computed from 2D canvas position via camera unprojection

**MouseButton**: Enumeration of mouse buttons (LEFT, RIGHT, MIDDLE, etc.)

Examples
--------
Set a custom event filter on a view::

    from scenex.model import View
    from scenex.app.events import MousePressEvent


    def on_click(event):
        if isinstance(event, MousePressEvent):
            print(f"Clicked at {event.canvas_pos}")
            return True  # Event handled
        return False


    view = View(scene=my_scene, camera=my_camera)
    view.set_event_filter(on_click)

See Also
--------
scenex.model.Camera : Camera with interactive controllers
scenex.model.View : View with event filter support
"""

from ._events import (
    Event,
    EventFilter,
    MouseButton,
    MouseDoublePressEvent,
    MouseEnterEvent,
    MouseEvent,
    MouseLeaveEvent,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
    Ray,
    ResizeEvent,
    WheelEvent,
)

__all__ = [
    "Event",
    "EventFilter",
    "MouseButton",
    "MouseDoublePressEvent",
    "MouseEnterEvent",
    "MouseEvent",
    "MouseLeaveEvent",
    "MouseMoveEvent",
    "MousePressEvent",
    "MouseReleaseEvent",
    "Ray",
    "ResizeEvent",
    "WheelEvent",
]
