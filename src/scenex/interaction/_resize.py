"""Resize policies for adapting view projections to canvas size changes."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

from pydantic import Field, PrivateAttr

from scenex.model._base import EventedBase

if TYPE_CHECKING:
    from scenex.model._transform import Transform
    from scenex.model._view import View


class ResizePolicy(EventedBase):
    """Base class defining how a view adapts to changes in its layout dimensions.

    A ResizePolicy is invoked automatically when a view's layout dimensions change,
    providing a hook to adjust any aspect of the view in response. Policies are
    registered with a CanvasInteractor and called whenever the layout width or height
    changes.

    Examples
    --------
    Register with CanvasInteractor::

        ci = CanvasInteractor(canvas)
        ci.set_resize_policy(view, Letterbox())

    See Also
    --------
    Letterbox : Resize policy that maintains aspect ratio
    CanvasInteractor : Coordinator that manages resize policies
    """

    @abstractmethod
    def handle_resize(self, view: View) -> None:
        """Respond to view layout dimension changes.

        Parameters
        ----------
        view : View
            The view being resized.
        """
        raise NotImplementedError


class Letterbox(ResizePolicy):
    """Maintain content aspect ratio on resize via letterboxing/pillarboxing.

    The Letterbox policy preserves the original aspect ratio of the camera's
    projection when the view is resized. When the view's aspect ratio differs from
    the content's aspect ratio, the projection is expanded in the narrower dimension
    to ensure all original content remains visible with black bars.

    Examples
    --------
    Register with CanvasInteractor::

        ci = CanvasInteractor(canvas)
        ci.set_resize_policy(view, Letterbox())

    See Also
    --------
    ResizePolicy : Base class for resize policies
    CanvasInteractor : Coordinator that manages resize policies
    """

    _reference: Transform | None = PrivateAttr(default=None)
    _last_adjustment: Transform | None = PrivateAttr(default=None)

    type: Literal["letterbox"] = Field(default="letterbox", repr=False)

    def handle_resize(self, view: View) -> None:
        """Handle view resize by adjusting projection to maintain aspect ratio."""
        if view.camera.projection != self._last_adjustment or self._reference is None:
            self._reference = view.camera.projection

        if (view_rect := view.rect) is None or self._reference is None:
            return
        _, _, view_width, view_height = view_rect
        if view_height == 0:
            return

        ref_mat = self._reference.root
        ref_x_scale = ref_mat[0, 0]
        ref_y_scale = ref_mat[1, 1]
        if ref_y_scale == 0:
            return

        view_aspect = view_width / view_height
        content_aspect = abs(ref_y_scale / ref_x_scale)

        if content_aspect < view_aspect:
            adjusted_proj = self._reference.scaled(
                (content_aspect / view_aspect, 1.0, 1.0)
            )
        else:
            adjusted_proj = self._reference.scaled(
                (1.0, view_aspect / content_aspect, 1.0)
            )

        view.camera.projection = self._last_adjustment = adjusted_proj


AnyResizePolicy = Letterbox | None
