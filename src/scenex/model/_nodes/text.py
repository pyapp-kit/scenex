from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from cmap import Color
from pydantic import Field, computed_field

from .node import AABB, Node

if TYPE_CHECKING:
    from scenex.app.events._events import Ray


class Text(Node):
    """A text label positioned in 3D world space.

    The text maintains a constant screen size regardless of camera zoom or distance,
    making it useful for labels, annotations, and markers. The text is positioned at the
    node's transformed origin point.

    Attributes
    ----------
    text : str
        The string content to display.
    color : Color
        Color of the text. Default is white.
    size : int
        Font size in pixels. Must be non-negative.

    Examples
    --------
    Create a simple text label:
        >>> text = Text(text="Hello World", color=Color("white"), size=14)

    Notes
    -----
    Text maintains constant screen size, not world size. The font size is specified in
    pixels and does not scale with camera zoom or distance from the viewer.
    """

    node_type: Literal["text"] = "text"

    text: str = Field(default="", description="The string content to display")
    color: Color = Field(default=Color("white"), description="Color of the text")
    size: int = Field(default=12, ge=0, description="Font size in pixels")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bounding_box(self) -> AABB:
        # TODO: Bounding boxes for text are hard.
        # First, they depend on the font, size, and actual text.
        # Second, text is completely virtual - its "size" varies as a
        # function of the camera to maintain size in the canvas space.
        #
        # Theoretically, we could compute a bounding box in screen space (leaning on
        # FreeType) and then unproject that back into world space, but that seems
        # overly complicated for now.
        #
        # Let's just return a point bounding box for now.
        return ((-1e-6, -1e-6, -1e-6), (1e-6, 1e-6, 1e-6))

    def passes_through(self, ray: Ray) -> float | None:
        # TODO: This faces similar issues to the bounding box problem.
        # Theoretically, we could compute intersection in canvas space.
        return None
