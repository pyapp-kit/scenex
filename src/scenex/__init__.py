"""Declarative scene graph model."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scenex")
except PackageNotFoundError:
    __version__ = "uninstalled"

from .model._canvas import Canvas
from .model._nodes.camera import Camera
from .model._nodes.image import Image
from .model._nodes.node import Node
from .model._nodes.points import Points
from .model._nodes.scene import Scene
from .model._nodes.volume import Volume
from .model._transform import Transform
from .model._view import View
from .util import loop, show, use

__all__ = [
    "Camera",
    "Canvas",
    "Image",
    "Node",
    "Points",
    "Scene",
    "Transform",
    "View",
    "Volume",
    "loop",
    "show",
    "use",
]
