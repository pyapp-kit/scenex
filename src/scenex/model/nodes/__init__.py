"""Nodes are world objects that can be placed in a scene.

`Scene` is a special node that represents the root of the scene graph.
"""

from .node import Node  # noqa: I001  must be imported first to avoid circular imports
from .camera import Camera
from .image import Image
from .points import Points
from .scene import Scene


__all__ = ["Camera", "Image", "Node", "Points", "Scene"]
