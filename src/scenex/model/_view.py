"""View model and controller classes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from pydantic import ConfigDict, Field, PrivateAttr

from ._base import EventedBase
from ._layout import Layout
from ._nodes.camera import Camera
from ._nodes.scene import Scene

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from scenex.adaptors._base import ViewAdaptor
    from scenex.app.events import Event

    from ._canvas import Canvas

logger = logging.getLogger(__name__)


class View(EventedBase):
    """An association of a scene and a camera.

    A view represents a rectangular area on a canvas that displays a single scene with a
    single camera.

    A canvas can have one or more views. Each view has a single scene (i.e. a
    scene graph of nodes) and a single camera. The camera defines the view
    transformation.  This class just exists to associate a single scene and
    camera.
    """

    scene: Scene = Field(default_factory=Scene)
    camera: Camera = Field(default_factory=Camera)
    layout: Layout = Field(default_factory=Layout, frozen=True)
    visible: bool = Field(default=True, description="Whether the view is visible.")

    _canvas: Canvas | None = PrivateAttr(None)

    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook for the model."""
        super().model_post_init(__context)
        self.camera.parent = self.scene

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
