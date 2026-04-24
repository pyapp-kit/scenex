"""Interaction module for scenex.

Provides camera controllers, resize policies, and the CanvasInteractor coordinator
that routes events through a well-defined pipeline.

Event Pipeline (in order; stops at the first step that returns True):
  1. Canvas-level event filter (via CanvasInteractor.set_event_filter)
  2. Per-view ResizePolicy (via CanvasInteractor.set_resize_policy)
  3. Per-view event filter (via CanvasInteractor.set_view_filter)
  4. Per-view CameraController (via CanvasInteractor.set_controller)

Examples
--------
Attach a pan/zoom controller and letterbox resize to a view::

    from scenex.interaction import CanvasInteractor, PanZoom, Letterbox

    ci = CanvasInteractor(canvas)
    ci.set_controller(view, PanZoom())
    ci.set_resize_policy(view, Letterbox())

Or use the per-view convenience wrapper::

    from scenex.interaction import ViewInteractor, PanZoom, Letterbox

    vi = ViewInteractor(view, controller=PanZoom(), resize_policy=Letterbox())
"""

from scenex.interaction._controllers import (
    AnyController,
    CameraController,
    Orbit,
    PanZoom,
)
from scenex.interaction._coordinator import CanvasInteractor, ViewInteractor
from scenex.interaction._resize import AnyResizePolicy, Letterbox, ResizePolicy

__all__ = [
    "AnyController",
    "AnyResizePolicy",
    "CameraController",
    "CanvasInteractor",
    "Letterbox",
    "Orbit",
    "PanZoom",
    "ResizePolicy",
    "ViewInteractor",
]
