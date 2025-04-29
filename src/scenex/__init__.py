"""Declarative scene graph model."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scenex")
except PackageNotFoundError:
    __version__ = "uninstalled"

from .model.canvas import Canvas
from .model.nodes.camera import Camera
from .model.nodes.image import Image
from .model.nodes.node import Node
from .model.nodes.points import Points
from .model.nodes.scene import Scene
from .model.transform import Transform
from .model.view import View
from .util import show

__all__ = [
    "Camera",
    "Canvas",
    "Image",
    "Node",
    "Points",
    "Scene",
    "Transform",
    "View",
    "show",
]
