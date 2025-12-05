"""Declarative scene graph model."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scenex")
except PackageNotFoundError:
    __version__ = "uninstalled"

from .adaptors import run, use
from .model._canvas import Canvas
from .model._color import ColorModel
from .model._controller import (
    LetterboxResizeStrategy,
    OrbitController,
    PanZoomController,
)
from .model._grid import Grid
from .model._nodes.camera import Camera
from .model._nodes.image import Image
from .model._nodes.line import Line
from .model._nodes.mesh import Mesh
from .model._nodes.node import Node
from .model._nodes.points import Points
from .model._nodes.scene import Scene
from .model._nodes.text import Text
from .model._nodes.volume import Volume
from .model._transform import Transform
from .model._view import View
from .util import show

__all__ = [
    "Camera",
    "Canvas",
    "ColorModel",
    "Grid",
    "Image",
    "LetterboxResizeStrategy",
    "Line",
    "Mesh",
    "Node",
    "OrbitController",
    "PanZoomController",
    "Points",
    "Scene",
    "Text",
    "Transform",
    "View",
    "Volume",
    "run",
    "show",
    "use",
]
