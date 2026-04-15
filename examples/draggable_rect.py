"""Draggable rectangle with corner handles overlaid on a grayscale image.

This example demonstrates:
- Composing a rectangle from three nodes: a semi-transparent mesh (fill),
  a solid line (outline), and square point markers (handles).
- Using an event filter to detect which handle is under the cursor and
  changing the cursor shape to signal that the handle is draggable.
- Keeping the dragged handle pinned to the cursor while the rest of the
  rectangle updates to stay consistent.
- Consuming mouse events during a drag so the camera does not also pan.
"""

import cmap
import numpy as np

import scenex as snx
from scenex.app import CursorType, app
from scenex.app.events import (
    Event,
    MouseButton,
    MouseLeaveEvent,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)

snx.use("vispy")

# ── Background image ──────────────────────────────────────────────────────────
IMG_W, IMG_H = 100, 100
rng = np.random.default_rng(0)

xs = np.linspace(0, 3 * np.pi, IMG_W)
ys = np.linspace(0, 3 * np.pi, IMG_H)
xx, yy = np.meshgrid(xs, ys)
base = ((np.sin(xx) * np.cos(yy) * 0.5 + 0.5) * 180 + 40).astype(np.float32)
noise = rng.integers(-12, 12, base.shape, dtype=np.int16)
img_data = np.clip(base + noise, 0, 255).astype(np.uint8)

bg = snx.Image(
    data=img_data,
    cmap=cmap.Colormap("grays"),
    clims=(0, 255),
    order=6,
)

# ── Rectangle nodes (line and handles are children of mesh, sharing its transform) ──
RECT_VERTICES = np.array(
    [[40, 40, 0], [60, 40, 0], [60, 60, 0], [40, 60, 0]], dtype=float
)

rect_mesh = snx.Mesh(
    vertices=RECT_VERTICES,
    faces=np.array([[0, 1, 2], [0, 2, 3]]),
    color=snx.UniformColor(color=cmap.Color("royalblue")),
    opacity=0.25,
    order=5,
)

rect_line = snx.Line(
    parent=rect_mesh,
    vertices=RECT_VERTICES[[0, 1, 2, 3, 0]],
    color=snx.UniformColor(color=cmap.Color("white")),
    width=2.0,
    order=4,
)

handles = snx.Points(
    parent=rect_mesh,
    vertices=RECT_VERTICES,
    size=14,
    face_color=snx.UniformColor(color=cmap.Color("white")),
    edge_color=snx.UniformColor(color=cmap.Color("royalblue")),
    edge_width=3,
    symbol="star",
    scaling="fixed",
    order=3,
)

# ── Scene / View ──────────────────────────────────────────────────────────────
scene = snx.Scene(children=[bg, rect_mesh])
view = snx.View(
    scene=scene,
    camera=snx.Camera(controller=snx.PanZoom(), interactive=True),
)
canvas = snx.show(view)

# ── Drag state ────────────────────────────────────────────────────────────────
_anchor: tuple[float, float] | None = None  # resize: fixed opposite corner
_drag_start: np.ndarray | None = None  # translate: cursor pos last frame


def _cursor_for_pos(wx: float, wy: float) -> CursorType:
    # Even corners (BL=0, TR=2) are on the main diagonal
    # Odd corners (BR=1, TL=3) on the anti-diagonal.
    return (
        CursorType.BDIAG_ARROW
        if _nearest_corner(wx, wy) % 2 == 0
        else CursorType.FDIAG_ARROW
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _new_vertices(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    mi = (min(x0, x1), min(y0, y1))
    ma = (max(x0, x1), max(y0, y1))
    return np.asarray(
        [
            (mi[0], mi[1], 0),
            (mi[0], ma[1], 0),
            (ma[0], ma[1], 0),
            (ma[0], mi[1], 0),
        ]
    )


def _nearest_corner(wx: float, wy: float) -> int:
    """Index of the corner handle nearest to world position (wx, wy)."""
    world = rect_mesh.transform.map(rect_mesh.vertices)[:, :2]
    return int(np.argmin(np.linalg.norm(world - [wx, wy], axis=1)))


# ── Event filter ──────────────────────────────────────────────────────────────


def _event_filter(event: Event) -> bool:
    global _anchor, _drag_start

    if isinstance(event, MouseMoveEvent):
        pos = np.array(event.world_ray.origin[:2])

        # -- Dragging a handle -- #
        if _anchor is not None:
            # Determine the new rectangle bounding box from the anchor point and the
            # mouse position
            new_vertices = _new_vertices(pos[0], pos[1], *_anchor)
            # Update the nodes
            rect_mesh.vertices = new_vertices
            rect_line.vertices = new_vertices[[0, 1, 2, 3, 0]]
            handles.vertices = new_vertices
            # Reset the cursor in case we are "flipping" the rectangle
            # by dragging a corner past an edge it is not connected to.
            app().set_cursor(canvas, _cursor_for_pos(*pos))
            return True

        # -- Dragging the whole rectangle -- #
        if _drag_start is not None:
            # Determine the offset since the last mouse event
            delta = pos - _drag_start
            # Offset two vertices to get the new rectangle bounding box
            # NOTE we just need two opposite corners, doesn't matter which two.
            v0 = rect_mesh.vertices[0, :2] + delta
            v2 = rect_mesh.vertices[2, :2] + delta
            new_vertices = _new_vertices(*v0, *v2)
            # Update the nodes
            rect_mesh.vertices = new_vertices
            rect_line.vertices = new_vertices[[0, 1, 2, 3, 0]]
            handles.vertices = new_vertices
            # Record the new position for a future drag
            _drag_start = pos
            return True

        # -- Hover -- #
        if event.world_ray.intersections(handles):
            app().set_cursor(canvas, _cursor_for_pos(*pos))
        elif event.world_ray.intersections(rect_mesh):
            app().set_cursor(canvas, CursorType.ALL_ARROW)
        else:
            app().set_cursor(canvas, CursorType.DEFAULT)

    elif isinstance(event, MousePressEvent):
        if event.buttons & MouseButton.LEFT:
            pos = np.array(event.world_ray.origin[:2])
            # -- Start a handle drag -- #
            if event.world_ray.intersections(handles):
                clicked = _nearest_corner(*pos)
                opp = (clicked + 2) % 4
                _anchor = rect_mesh.vertices[opp, :2]
                return True
            # -- Start a rectangle drag -- #
            if event.world_ray.intersections(rect_mesh):
                _drag_start = pos
                return True

    elif isinstance(event, MouseReleaseEvent):
        # -- End a drag -- #
        _anchor = None
        _drag_start = None
        return True

    elif isinstance(event, MouseLeaveEvent):
        # -- End a drag -- #
        _anchor = None
        _drag_start = None
        app().set_cursor(canvas, CursorType.DEFAULT)

    return False


view.set_event_filter(_event_filter)
snx.run()
