from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pylinalg as la
from cmap import Color
from pydantic import ConfigDict, Field

from scenex.app.events import Event, MouseEvent, Ray, ResizeEvent

from ._base import EventedBase
from ._evented_list import EventedList
from ._view import View  # noqa: TC001

if TYPE_CHECKING:
    from scenex import Node
    from scenex.adaptors._base import CanvasAdaptor


class Canvas(EventedBase):
    """Canvas onto which views are rendered.

    In desktop applications, this will be a window. In web applications, this will be a
    div.  The canvas has one or more views, which are rendered onto it.  For example,
    an orthoviewer might be a single canvas with three views, one for each axis.
    """

    width: int = Field(default=500, description="The width of the canvas in pixels.")
    height: int = Field(default=500, description="The height of the canvas in pixels.")
    background_color: Color = Field(
        default=Color("black"), description="The background color."
    )
    visible: bool = Field(default=False, description="Whether the canvas is visible.")
    title: str = Field(default="", description="The title of the canvas.")
    views: EventedList[View] = Field(default_factory=EventedList, frozen=True)

    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook for the model."""
        # Update all current views
        for view in self.views:
            view._canvas = self
        # Update all views added later
        self.views.item_inserted.connect(self._on_view_inserted)
        self.views.item_changed.connect(self._on_view_changed)
        self.views.item_removed.connect(self._on_view_removed)

        self.events.width.connect(self._recompute_layout)
        self.events.height.connect(self._recompute_layout)

        self._recompute_layout()

    def _recompute_layout(self, dont_use: int | None = None) -> None:
        if not len(self.views):
            # Nothing to do
            return
        # The parameter is EITHER width or height - just use the model values instead
        width, height = self.size
        # FIXME: Allow customization
        x = 0.0
        dx = float(width) / len(self.views)

        for view in self.views:
            view.layout.x = x
            view.layout.y = 0
            view.layout.width = dx
            view.layout.height = height
            x += dx

    def _on_view_inserted(self, idx: int, view: View) -> None:
        view._canvas = self
        self._recompute_layout()

    def _on_view_removed(self, idx: int, view: View) -> None:
        view._canvas = None
        self._recompute_layout()

    def _on_view_changed(
        self,
        idx: int | slice,
        old_view: View | Sequence[View],
        new_view: View | Sequence[View],
    ) -> None:
        if not isinstance(old_view, Sequence):
            old_view = [old_view]
        for view in old_view:
            view._canvas = None

        if not isinstance(new_view, Sequence):
            new_view = [new_view]
        for view in new_view:
            view._canvas = self
        self._recompute_layout()

    @property
    def size(self) -> tuple[int, int]:
        """Return the size of the canvas."""
        return self.width, self.height

    @size.setter
    def size(self, value: tuple[int, int]) -> None:
        """Set the size of the canvas."""
        self.width, self.height = value

    def render(self) -> np.ndarray:
        """Show the canvas."""
        if adaptors := self._get_adaptors():
            return cast("CanvasAdaptor", adaptors[0])._snx_render()
        raise RuntimeError("No adaptor found for Canvas.")

    def handle(self, event: Event) -> bool:
        """Handle the passed event."""
        handled = False
        if isinstance(event, MouseEvent):
            if view := self._containing_view(event.canvas_pos):
                through: list[tuple[Node, float]] = []
                for child in view.scene.children:
                    if (d := child.passes_through(event.world_ray)) is not None:
                        through.append((child, d))

                # FIXME: Consider only reporting the first?
                # Or do we only report until we hit a node with opacity=1?
                for node, _depth in sorted(through, key=lambda e: e[1]):
                    # Filter through parent scenes to child
                    handled |= Canvas._filter_through(event, node, node)
                # No nodes in the view handled the event - pass it to the camera
                if not handled and view.camera.interactive:
                    handled |= view.camera.filter_event(event, view.camera)
        elif isinstance(event, ResizeEvent):
            # TODO: How might some event filter tap into the resize?
            self.size = (event.width, event.height)
        return handled

    @staticmethod
    def _filter_through(event: Any, node: Node, target: Node) -> bool:
        """Filter the event through the scene graph to the target node."""
        # TODO: Suppose a scene is not interactive. If the node is interactive, should
        # it receive the event?

        # First give this node a chance to filter the event.

        if node.interactive and node.filter_event(event, target):
            # Node filtered out the event, so we stop here.
            return True
        if (parent := node.parent) is None:
            # Node did not filter out the event, and we've reached the top of the graph.
            return False
        # Recursively filter the event through node's parent.
        return Canvas._filter_through(event, parent, target)

    def to_world(self, canvas_pos: tuple[float, float]) -> Ray | None:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""
        # Code adapted from:
        # https://github.com/pygfx/pygfx/pull/753/files#diff-173d643434d575e67f8c0a5bf2d7ea9791e6e03a4e7a64aa5fa2cf4172af05cdR395
        view = self._containing_view(canvas_pos)
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
        # Unproject the canvas NDC coordinates into world space.
        pos_world = la.vec_unproject(pos_ndc, camera_matrix)

        # To find the direction of the ray, we find a unprojected point farther away
        # and subtract the closer point.
        pos_world_farther = la.vec_unproject(pos_ndc, camera_matrix, depth=1)
        direction = pos_world_farther - pos_world
        direction = direction / np.linalg.norm(direction)

        ray = Ray(
            origin=tuple(pos_world),
            direction=tuple(direction),
        )
        return ray

    def _containing_view(self, pos: tuple[float, float]) -> View | None:
        for view in self.views:
            if pos in view.layout:
                return view
        return None
