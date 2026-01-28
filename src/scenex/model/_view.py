"""View model and resize strategies."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, cast

from pydantic import Field, PrivateAttr

from ._base import EventedBase
from ._layout import Layout
from ._nodes.camera import Camera
from ._nodes.scene import Scene

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from scenex import Transform
    from scenex.adaptors._base import ViewAdaptor
    from scenex.app.events import Event

    from ._canvas import Canvas

logger = logging.getLogger(__name__)

AnyResizeStrategy = Annotated[Union["Letterbox", "None"], Field(discriminator="type")]


class View(EventedBase):
    """A rectangular viewport that displays a scene through a camera.

    A View represents a rectangular area on a canvas that renders a scene graph through
    a specific camera perspective. Each view associates exactly one scene with one
    camera, defining what is displayed and how it is viewed. Multiple views can exist
    on a single canvas, each potentially showing different scenes or the same scene from
    different camera angles.

    Attributes
    ----------
    scene : Scene
        The scene graph containing all visual elements to be rendered in this view.
    camera : Camera
        The camera defining the viewing perspective and projection for this view.
    resize : ResizeStrategy | None
        Strategy for adjusting the camera projection when the view is resized. If None,
        the camera projection remains unchanged on view resize.
    layout : Layout
        The layout defining the view's position, size, and visual styling on the canvas.
    visible : bool
        Whether the view is visible and should be rendered.

    Examples
    --------
    Create a view with a scene containing an image:
        >>> import numpy as np
        >>> my_array = np.random.rand(100, 100).astype(np.float32)
        >>> scene = Scene(children=[Image(data=my_array)])
        >>> view = View(scene=scene, camera=Camera())

    Create a view with interactive camera and letterbox resizing:
        >>> view = View(
        ...     scene=scene,
        ...     camera=Camera(controller=PanZoom(), interactive=True),
        ...     resize=Letterbox(),
        ... )

    Add a view to a canvas:
        >>> canvas = Canvas()
        >>> canvas.views.append(view)
    """

    scene: Scene = Field(
        default_factory=Scene,
        description="The scene graph containing all visual elements to render",
    )
    camera: Camera = Field(
        default_factory=Camera,
        description="The camera defining the viewing perspective and projection",
    )
    resize: AnyResizeStrategy = Field(
        default=None,
        description="Strategy for adjusting camera projection when the view is resized",
    )
    layout: Layout = Field(
        default_factory=Layout,
        frozen=True,
        description="The layout defining position, size, and visual styling",
    )
    visible: bool = Field(
        default=True, description="Whether the view is visible and should be rendered"
    )

    _canvas: Canvas | None = PrivateAttr(None)

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook for the model."""
        super().model_post_init(__context)
        self.camera.parent = self.scene

        # FIXME: Reconnect this when the layout is changed
        self.layout.events.width.connect(self._on_layout_change)
        self.layout.events.height.connect(self._on_layout_change)

    def _on_layout_change(self, *args: Any) -> None:
        if resize := self.resize:
            resize.handle_resize(self)

    @property
    def canvas(self) -> Canvas:
        """The canvas that the view is on.

        If one hasn't been created/assigned, a new one is created.
        """
        if (canvas := self._canvas) is None:
            from ._canvas import Canvas

            self.canvas = canvas = Canvas()
        return canvas

    @canvas.setter
    def canvas(self, value: Canvas) -> None:
        self._canvas = value
        # If this view is not already on the canvas, just add it to the end
        if self not in value.views:
            value.views.append(self)

    def render(self) -> np.ndarray:
        """Render the view to an array."""
        if adaptors := self._get_adaptors():
            return cast("ViewAdaptor", adaptors[0])._snx_render()
        raise RuntimeError("No adaptor found for View.")

    _filter: Callable[[Event], bool] | None = PrivateAttr(default=None)

    def set_event_filter(
        self, callable: Callable[[Event], bool] | None
    ) -> Callable[[Event], bool] | None:
        """
        Registers a callable to filter events.

        Parameters
        ----------
        callable : Callable[[Event], bool] | None
            A callable that takes an Event and returns True if the event was handled,
            False otherwise. Passing None is equivalent to removing any existing filter.
            By returning True, the callable indicates that the event has been handled
            and should not be propagated to subsequent handlers.

        Returns
        -------
        Callable[[Event], bool] | None
            The previous event filter, or None if there was no filter.

        Note the name has parity with Node.filter_event, but there's not much filtering
        going on.
        """
        old, self._filter = self._filter, callable
        return old

    def filter_event(self, event: Event) -> bool:
        """
        Filters the event.

        This method allows the larger view to react to events that:
        1. Require summarization of multiple smaller event responses.
        2. Could not be picked up by a node (e.g. mouse leaving an image).

        Note the name has parity with Node.filter_event, but there's not much filtering
        going on.

        Parameters
        ----------
        event : Event
            An event occurring in the view.

        Returns
        -------
        bool
            True iff the event should not be propagated to other handlers.
        """
        if self._filter:
            handled = self._filter(event)
            if not isinstance(handled, bool):
                # Some widget frameworks (i.e. Qt) get upset when non-booleans are
                # returned. If the event-filter does not return a boolean, rather than
                # letting that propagate upwards, we log a warning and return False.
                logger.warning(
                    f"Event filter {self._filter} did not return a boolean. "
                    "Returning False."
                )
                # Return False. We assume that if the user wanted to block future
                # processing, they'd be less likely to forget a boolean return.
                # Further, allowing downstream processing is a clear sign to they author
                # that they forgot to block propagation.
                handled = False
            return handled
        return False


# ====================================================================================
# Resize Strategies
# ====================================================================================


class ResizeStrategy(EventedBase):
    """Base class defining how a view adapts to changes in its layout dimensions.

    A ResizeStrategy is invoked automatically when a view's layout dimensions change,
    providing a hook to adjust any aspect of the view in response. While the most
    common use case is adjusting the camera's projection matrix to maintain aspect
    ratio or fit content, strategies have full access to the view and can modify the
    camera, scene, layout, or any other properties as needed.

    Strategies are attached to View instances and called whenever the layout width
    or height changes, whether from user interaction (window resize, splitter drag)
    or programmatic updates.

    Examples
    --------
    Maintain aspect ratio when view resizes:
        >>> view = View(camera=Camera(), resize=Letterbox())

    No resize behavior (omit the resize parameter):
        >>> view = View(camera=Camera())

    See Also
    --------
    Letterbox : Resize strategy that maintains aspect ratio
    View : View class that uses resize strategies
    Camera : Camera class with projection property
    """

    @abstractmethod
    def handle_resize(self, view: View) -> None:
        """
        Respond to view layout dimension changes.

        This method is called automatically when the view's layout dimensions change.
        Implementations have full access to the view and can modify any of its
        properties.

        Parameters
        ----------
        view : View
            The view being resized.
        """
        raise NotImplementedError


class Letterbox(ResizeStrategy):
    """Maintain content aspect ratio on resize via letterboxing/pillarboxing.

    The Letterbox strategy preserves the original aspect ratio of the camera's
    projection when the view is resized. When the view's aspect ratio differs from
    the content's aspect ratio, the projection is expanded in the narrower dimension
    to ensure all original content remains visible with black bars (letterboxing for
    wide views, pillarboxing for tall views).

    The strategy tracks resize sequences (e.g., dragging a window corner) by storing
    the camera's projection as a reference at the start of that sequence. At any point
    during the sequence, the projection matrix is expanded in either width or height to
    retain the rectangle of that reference projection. A new sequence is defined by a
    change in the projection matrix, either programmatically made or through user input,
    signalled by a camera projection matrix different from that set during the last
    resize operation.

    Examples
    --------
    Create a view with letterbox resizing:
        >>> from scenex.utils.projections import orthographic
        >>> view = View(
        ...     camera=Camera(projection=orthographic(100, 100, 100)),
        ...     resize=Letterbox(),
        ... )

    When view is resized to 200x100 pixels, the projection expands horizontally
    to maintain the 1:1 aspect ratio, showing more content on the sides rather
    than stretching the image.

    Notes
    -----
    This approach follows the conventions of vispy's PanZoomCamera and pygfx's
    PerspectiveCamera. The projection matrix scales are inversely proportional to
    the displayed region: smaller scale values show more content.

    See Also
    --------
    ResizeStrategy : Base class for resize strategies
    View : View class that uses resize strategies
    Camera : Camera class with projection property
    """

    # Consider the context of a sequence of resizes (i.e. the user is clicking and
    # dragging the window corner).
    # This is the transform at the beginning of the resize sequence...
    _reference: Transform | None = PrivateAttr(default=None)
    # ...and this is the transform we applied in response to the last resize event.
    _last_adjustment: Transform | None = PrivateAttr(default=None)

    type: Literal["letterbox"] = Field(default="letterbox", repr=False)

    def handle_resize(self, view: View) -> None:
        """Handle view resize by adjusting projection to maintain aspect ratio."""
        # If the current projection differs from the last adjustment, or if there is no
        # reference to begin with, this is a new resize sequence.
        if view.camera.projection != self._last_adjustment or self._reference is None:
            self._reference = view.camera.projection

        view_width = int(view.layout.width)
        view_height = int(view.layout.height)
        if view_height == 0 or self._reference is None:
            return

        # Extract projection scales that define the content aspect ratio
        ref_mat = self._reference.root
        ref_x_scale = ref_mat[0, 0]
        ref_y_scale = ref_mat[1, 1]
        if ref_y_scale == 0:
            return

        # Compute aspect ratios
        # NOTE: projection scales are inversely proportional to the displayed region,
        # so content_aspect = y_scale / x_scale
        view_aspect = view_width / view_height
        content_aspect = abs(ref_y_scale / ref_x_scale)

        # Expand the narrower dimension to match the view aspect
        if content_aspect < view_aspect:
            # View is wider: expand horizontal frustum (reduce x scale)
            adjusted_proj = self._reference.scaled(
                (content_aspect / view_aspect, 1.0, 1.0)
            )
        else:
            # View is taller: expand vertical frustum (reduce y scale)
            adjusted_proj = self._reference.scaled(
                (1.0, view_aspect / content_aspect, 1.0)
            )

        # Store the adjustment before applying it
        view.camera.projection = self._last_adjustment = adjusted_proj
