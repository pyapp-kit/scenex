"""CanvasInteractor and ViewInteractor: coordinators for interaction components."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

from scenex.app.events import MouseEnterEvent, MouseEvent, MouseLeaveEvent

if TYPE_CHECKING:
    from collections.abc import Callable

    from scenex.app.events import Event
    from scenex.interaction._controllers import CameraController
    from scenex.interaction._resize import ResizePolicy
    from scenex.model._canvas import Canvas
    from scenex.model._view import View

logger = logging.getLogger("scenex.interaction")

# Global weak-value registry: maps canvas model_id (hex str) → CanvasInteractor.
# Canvas adaptors look up this registry in their _dispatch_event method so that
# event routing works regardless of whether the interactor was created before or
# after the adaptor.
_interactor_by_canvas_id: WeakValueDictionary[str, CanvasInteractor] = (
    WeakValueDictionary()
)


@dataclass
class _ViewState:
    """Per-view interaction state owned by a CanvasInteractor."""

    controller: CameraController | None = None
    resize_policy: ResizePolicy | None = None
    event_filter: Callable[[Event], bool] | None = None
    # (signal, slot) pairs for resize signal subscriptions
    _resize_connections: list = field(default_factory=list)

    def disconnect_resize(self) -> None:
        for sig, slot in self._resize_connections:
            try:
                sig.disconnect(slot)
            except Exception:
                pass
        self._resize_connections.clear()


class CanvasInteractor:
    """Coordinator that manages interaction components for a Canvas.

    ``CanvasInteractor`` is the single entry point for all events on a canvas.
    It owns the ordered event pipeline and manages per-view interaction state
    (controllers, resize policies, and event filters).

    Event Pipeline (in order; stops at the first step that returns True):

    1. **Canvas-level event filter** — user-defined callable registered via
       ``set_event_filter()``.
    2. **ResizePolicy** — for each view, the registered resize policy is called
       when the view's dimensions change (via signals) or on a ResizeEvent.
    3. **View-level event filter** — per-view user-defined callable registered
       via ``set_view_filter()``.
    4. **CameraController** — per-view controller registered via
       ``set_controller()``.

    Examples
    --------
    Attach a pan/zoom controller and letterbox resize to a view::

        from scenex.interaction import CanvasInteractor, PanZoom, Letterbox

        ci = CanvasInteractor(canvas)
        ci.set_controller(view, PanZoom())
        ci.set_resize_policy(view, Letterbox())

    Register a canvas-level event filter::

        def my_filter(event):
            print(event)
            return False


        ci.set_event_filter(my_filter)
    """

    def __init__(self, canvas: Canvas) -> None:
        self._canvas = canvas
        self._event_filter: Callable[[Event], bool] | None = None
        self._view_states: dict[str, _ViewState] = {}  # keyed by view._model_id.hex
        self._last_mouse_view: View | None = None

        # Register in the global dict so canvas adaptors can find us.
        _interactor_by_canvas_id[canvas._model_id.hex] = self

        # Track views added/removed from the canvas so we can manage their state.
        canvas.views.item_inserted.connect(self._on_view_inserted)
        canvas.views.item_removed.connect(self._on_view_removed)

    # ------------------------------------------------------------------
    # Public configuration API
    # ------------------------------------------------------------------

    def set_event_filter(
        self, event_filter: Callable[[Event], bool] | None
    ) -> Callable[[Event], bool] | None:
        """Register a callable to filter all canvas events before view dispatch.

        Parameters
        ----------
        event_filter : Callable[[Event], bool] | None
            Callable that receives each Event and returns True if handled.
            Pass None to remove any existing filter.

        Returns
        -------
        Callable[[Event], bool] | None
            The previous event filter.
        """
        old, self._event_filter = self._event_filter, event_filter
        return old

    def set_controller(self, view: View, controller: CameraController | None) -> None:
        """Register a CameraController for a view.

        Parameters
        ----------
        view : View
            The view whose camera should be controlled.
        controller : CameraController | None
            The controller to use, or None to remove any existing controller.
        """
        self._get_or_create_state(view).controller = controller

    def set_resize_policy(self, view: View, policy: ResizePolicy | None) -> None:
        """Register a ResizePolicy for a view.

        Parameters
        ----------
        view : View
            The view whose projection should be adjusted on resize.
        policy : ResizePolicy | None
            The resize policy to use, or None to remove any existing policy.
        """
        state = self._get_or_create_state(view)
        state.disconnect_resize()
        state.resize_policy = policy

        if policy is not None:

            def _on_resize(*_: object) -> None:
                policy.handle_resize(view)

            for sig in (
                view.layout.events.x_start,
                view.layout.events.x_end,
                view.layout.events.y_start,
                view.layout.events.y_end,
            ):
                sig.connect(_on_resize)
                state._resize_connections.append((sig, _on_resize))

            if view.canvas is not None:
                for sig in (view.canvas.events.width, view.canvas.events.height):
                    sig.connect(_on_resize)
                    state._resize_connections.append((sig, _on_resize))

    def set_view_filter(
        self, view: View, event_filter: Callable[[Event], bool] | None
    ) -> Callable[[Event], bool] | None:
        """Register a per-view event filter.

        Parameters
        ----------
        view : View
            The view to attach the filter to.
        event_filter : Callable[[Event], bool] | None
            Callable that receives each Event routed to this view and returns
            True if handled. Pass None to remove any existing filter.

        Returns
        -------
        Callable[[Event], bool] | None
            The previous view filter.
        """
        state = self._get_or_create_state(view)
        old, state.event_filter = state.event_filter, event_filter
        return old

    # ------------------------------------------------------------------
    # Event pipeline entry point (called by canvas adaptors)
    # ------------------------------------------------------------------

    def handle(self, event: Event) -> bool:
        """Process an event through the interaction pipeline.

        Called by canvas adaptors for every backend event.

        Parameters
        ----------
        event : Event
            The event to process.

        Returns
        -------
        bool
            True if the event was handled and should not propagate further.
        """
        # Step 1: canvas-level event filter.
        if self._canvas_filter(event):
            return True

        # Steps 2-4 are per-view.
        if isinstance(event, MouseEvent):
            return self._handle_mouse_event(event)
        if isinstance(event, MouseLeaveEvent):
            return self._handle_mouse_leave(event)

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _canvas_filter(self, event: Event) -> bool:
        if self._event_filter is None:
            return False
        handled = self._event_filter(event)
        if not isinstance(handled, bool):
            logger.warning(
                "Canvas event filter %r did not return a bool; treating as False.",
                self._event_filter,
            )
            return False
        return handled

    def _handle_mouse_event(self, event: MouseEvent) -> bool:
        current_view = self._containing_view(event.pos)

        # Synthesize enter/leave transitions between views.
        if self._last_mouse_view != current_view:
            if self._last_mouse_view is not None:
                self._view_filter(self._last_mouse_view, MouseLeaveEvent())
            if current_view is not None and not isinstance(event, MouseEnterEvent):
                self._view_filter(
                    current_view,
                    MouseEnterEvent(pos=event.pos, buttons=event.buttons),
                )
        self._last_mouse_view = current_view

        if current_view is None:
            return False

        # Step 3: view-level event filter.
        if self._view_filter(current_view, event):
            return True

        # Step 4: camera controller.
        state = self._view_states.get(current_view._model_id.hex)
        if state and state.controller is not None:
            return state.controller.handle_event(event, current_view)

        return False

    def _handle_mouse_leave(self, event: MouseLeaveEvent) -> bool:
        if self._last_mouse_view is not None:
            handled = self._view_filter(self._last_mouse_view, event)
            self._last_mouse_view = None
            return handled
        return False

    def _view_filter(self, view: View, event: Event) -> bool:
        state = self._view_states.get(view._model_id.hex)
        if state is None or state.event_filter is None:
            return False
        handled = state.event_filter(event)
        if not isinstance(handled, bool):
            logger.warning(
                "View event filter %r did not return a bool; treating as False.",
                state.event_filter,
            )
            return False
        return handled

    def _containing_view(self, pos: tuple[float, float]) -> View | None:
        for view in self._canvas.views:
            if view.content_rect is None:
                continue
            x, y, w, h = view.content_rect
            if x <= pos[0] <= x + w and y <= pos[1] <= y + h:
                return view
        return None

    def _get_or_create_state(self, view: View) -> _ViewState:
        key = view._model_id.hex
        if key not in self._view_states:
            self._view_states[key] = _ViewState()
        return self._view_states[key]

    def _on_view_inserted(self, _: int, view: View) -> None:
        # If a resize policy is already registered for this view, reconnect
        # canvas size signals now that the view has a canvas.
        state = self._view_states.get(view._model_id.hex)
        if state and state.resize_policy is not None:
            self.set_resize_policy(view, state.resize_policy)

    def _on_view_removed(self, _: int, view: View) -> None:
        state = self._view_states.pop(view._model_id.hex, None)
        if state:
            state.disconnect_resize()


class ViewInteractor:
    """Convenience wrapper for single-view interaction configuration.

    ``ViewInteractor`` wraps a single ``View`` and configures a
    ``CanvasInteractor`` on the view's canvas. If the view is not yet attached
    to a canvas, the configuration is applied when it is.

    Examples
    --------
    ::

        from scenex.interaction import ViewInteractor, PanZoom, Letterbox

        vi = ViewInteractor(view, controller=PanZoom(), resize_policy=Letterbox())

    See Also
    --------
    CanvasInteractor : Lower-level coordinator for multi-view canvases
    """

    def __init__(
        self,
        view: View,
        *,
        controller: CameraController | None = None,
        resize_policy: ResizePolicy | None = None,
        event_filter: Callable[[Event], bool] | None = None,
    ) -> None:
        self._view = view
        self._pending_controller = controller
        self._pending_resize_policy = resize_policy
        self._pending_filter = event_filter
        self._canvas_interactor: CanvasInteractor | None = None

        if view.canvas is not None:
            self._attach(view.canvas)
        else:
            # Wait until the view is attached to a canvas.
            view.events.connect(self._on_view_event)

    def _attach(self, canvas: Canvas) -> None:
        ci = _interactor_by_canvas_id.get(canvas._model_id.hex)
        if ci is None:
            ci = CanvasInteractor(canvas)
        self._canvas_interactor = ci
        if self._pending_controller is not None:
            ci.set_controller(self._view, self._pending_controller)
        if self._pending_resize_policy is not None:
            ci.set_resize_policy(self._view, self._pending_resize_policy)
        if self._pending_filter is not None:
            ci.set_view_filter(self._view, self._pending_filter)

    def _on_view_event(self, _: object) -> None:
        # Re-check if view now has a canvas attached.
        if self._view.canvas is not None and self._canvas_interactor is None:
            self._attach(self._view.canvas)

    @property
    def canvas_interactor(self) -> CanvasInteractor | None:
        """The underlying CanvasInteractor, or None if the view has no canvas."""
        return self._canvas_interactor
