"""View model — a rectangular viewport displaying a scene through a camera."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pylinalg as la
from pydantic import Field, PrivateAttr

from scenex.app.events import Ray

from ._base import EventedBase
from ._layout import Layout
from ._nodes.camera import Camera
from ._nodes.scene import Scene

if TYPE_CHECKING:
    from scenex.adaptors._base import ViewAdaptor

    from ._canvas import Canvas

logger = logging.getLogger(__name__)


class View(EventedBase):
    """A rectangular viewport that displays a scene through a camera.

    A View represents a rectangular area on a canvas that renders a scene graph through
    a specific camera perspective. Each view associates exactly one scene with one
    camera, defining what is displayed and how it is viewed. Multiple views can exist
    on a single canvas, each potentially showing different scenes or the same scene from
    different camera angles.

    Examples
    --------
    Create a view with a scene containing an image:
        >>> import numpy as np
        >>> my_array = np.random.rand(100, 100).astype(np.float32)
        >>> scene = Scene(children=[Image(data=my_array)])
        >>> view = View(scene=scene, camera=Camera())

    Add a view to a canvas:
        >>> canvas = Canvas()
        >>> canvas.views.append(view)

    Attach interaction via CanvasInteractor::

        from scenex.interaction import CanvasInteractor, PanZoom, Letterbox

        ci = CanvasInteractor(canvas)
        ci.set_controller(view, PanZoom())
        ci.set_resize_policy(view, Letterbox())
    """

    scene: Scene = Field(
        default_factory=Scene,
        description="The scenegraph to render",
    )
    camera: Camera = Field(
        default_factory=Camera,
        description="The camera defining the viewing perspective and projection",
    )
    layout: Layout = Field(
        default_factory=Layout,
        frozen=True,
        description="Defines view position, size, and styling upon the canvas",
    )
    visible: bool = Field(
        default=True, description="Whether the view is visible and should be rendered"
    )

    # Backreference to the canvas displaying this view. Should not be set directly;
    # use the canvas property to ensure proper event connections.
    _canvas: Canvas | None = PrivateAttr(None)

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook for the model."""
        super().model_post_init(__context)
        self.camera.parent = self.scene

    @property
    def canvas(self) -> Canvas | None:
        """The canvas that the view is on."""
        return self._canvas

    @canvas.setter
    def canvas(self, value: Canvas | None) -> None:
        old, self._canvas = self._canvas, value

        if old:
            if self in old.views:
                old.views.remove(self)

        if self._canvas:
            if self not in self._canvas.views:
                self._canvas.views.append(self)

    @property
    def rect(self) -> tuple[int, int, int, int] | None:
        """Pixel rect (x, y, width, height) of this view on its canvas.

        None if the view is not on a canvas.
        """
        if self._canvas is not None:
            return self._canvas.rect_for(self)
        return None

    @property
    def content_rect(self) -> tuple[int, int, int, int] | None:
        """Pixel content rect (x, y, width, height) of this view, excluding insets.

        None if the view is not on a canvas.
        """
        if self._canvas is not None:
            return self._canvas.content_rect_for(self)
        return None

    def _to_ndc(self, view_pos: tuple[float, float]) -> tuple[float, float] | None:
        """Map a view-relative pixel position to normalized device coordinates (NDC)."""
        if (rect := self.content_rect) is None:
            return None
        _, _, width, height = rect
        ndc_x = view_pos[0] / width * 2 - 1
        ndc_y = -(view_pos[1] / height * 2 - 1)
        return (ndc_x, ndc_y)

    def to_ray(self, canvas_pos: tuple[float, float]) -> Ray | None:
        """Compute the world-space ray for a canvas position within this view.

        Parameters
        ----------
        canvas_pos : tuple[float, float]
            The (x, y) position in canvas pixel coordinates.

        Returns
        -------
        Ray | None
            The world-space Ray, or None if this view has no canvas.
        """
        if self._canvas is None:
            logger.warning(
                "to_ray() called on a View not attached to a Canvas. "
                "Canvas coordinates have no meaning without a canvas."
            )
            return None
        x, y = self._canvas.content_rect_for(self)[:2]
        view_pos = (canvas_pos[0] - x, canvas_pos[1] - y)
        ndc = self._to_ndc(view_pos)
        if ndc is None:
            return None
        return self._ndc_to_ray(ndc)

    def _ndc_to_ray(self, ndc: tuple[float, float]) -> Ray:
        """Unproject an NDC position to a world-space Ray through this view."""
        camera_matrix = self.camera.projection @ self.camera.transform.inv().T
        pos = la.vec_unproject(ndc, camera_matrix)
        pos_far = la.vec_unproject(ndc, camera_matrix, depth=1)
        direction = pos_far - pos
        direction = direction / np.linalg.norm(direction)
        return Ray(origin=tuple(pos), direction=tuple(direction), source=self)

    def render(self) -> np.ndarray:
        """Render the view to an array."""
        if adaptors := self._get_adaptors():
            return cast("ViewAdaptor", adaptors[0])._snx_render()
        raise RuntimeError("No adaptor found for View.")
