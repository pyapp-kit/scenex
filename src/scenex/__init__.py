"""Declarative scene graph library for scientific visualization.

scenex is a Python library for creating interactive 3D visualizations using a
declarative scene graph model. It provides a backend-agnostic API that works with
multiple rendering engines (pygfx, vispy) while maintaining a consistent, intuitive
interface.

Key Features
------------
- **Declarative API**: Describe how the scene should look rather than how to render it.
- **Evented models**: Events enable painless reaction to changes in the scene graph
- **Multiple backends**: Render with pygfx (WebGPU) or vispy (OpenGL)

Quick Start
-----------
Create and display a simple visualization::

    import numpy as np
    import scenex as snx

    # Create a random image
    data = np.random.rand(100, 100)
    img = snx.Image(data=data)

    # Show it
    snx.show(img)
    snx.run()

See Also
--------
- scenex.model: Core declarative model classes
- scenex.adaptors: Backend adaptor implementations
- scenex.app: Application and event handling
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scenex")
except PackageNotFoundError:
    __version__ = "uninstalled"

from .adaptors import run, use
from .model._canvas import Canvas
from .model._color import (
    ColorModel,
    FaceColors,
    UniformColor,
    VertexColors,
)
from .model._nodes.camera import Camera, CameraController, Orbit, PanZoom
from .model._nodes.image import Image
from .model._nodes.line import Line
from .model._nodes.mesh import Mesh
from .model._nodes.node import Node
from .model._nodes.points import Points
from .model._nodes.scene import Scene
from .model._nodes.text import Text
from .model._nodes.volume import Volume
from .model._transform import Transform
from .model._view import Letterbox, ResizePolicy, View
from .util import show

__all__ = [
    "Camera",
    "CameraController",
    "Canvas",
    "ColorModel",
    "FaceColors",
    "Image",
    "Letterbox",
    "Line",
    "Mesh",
    "Node",
    "Orbit",
    "PanZoom",
    "Points",
    "ResizePolicy",
    "Scene",
    "Text",
    "Transform",
    "UniformColor",
    "VertexColors",
    "View",
    "Volume",
    "run",
    "show",
    "use",
]
