"""View model and controller classes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr

from ._base import EventedBase
from .layout import Layout
from .nodes.camera import Camera
from .nodes.scene import Scene

if TYPE_CHECKING:
    from .canvas import Canvas

logger = logging.getLogger(__name__)


# just a random/basic selection of blend modes for now
BlendMode = Literal["default", "opaque", "alpha", "additive"]


class View(EventedBase):
    """A rectangular area on a canvas that displays a scene, with a camera.

    A canvas can have one or more views. Each view has a single scene (i.e. a
    scene graph of nodes) and a single camera. The camera defines the view
    transformation.  This class just exists to associate a single scene and
    camera.
    """

    scene: Scene = Field(default_factory=Scene)
    camera: Camera = Field(default_factory=Camera)
    layout: Layout = Field(default_factory=Layout, frozen=True)
    blending: BlendMode = Field(
        default="default",
        description="The blending mode to use when rendering the view. "
        "Must be one of 'default', 'opaque', 'alpha', or 'additive'.",
    )
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
            from .canvas import Canvas

            self.canvas = canvas = Canvas()
        return canvas

    @canvas.setter
    def canvas(self, value: Canvas) -> None:
        self._canvas = value
        self._canvas.views.append(self)
