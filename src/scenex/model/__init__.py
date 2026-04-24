"""Declarative model classes for building scene graphs.

This module provides all the core model objects for constructing interactive 3D
visualizations. The models form a scene graph - a hierarchical tree structure where
each node can have children, with transformations, visibility, and other properties
propagating down the tree.

Scene Graph Structure
---------------------
The scene graph follows this hierarchy::

    Canvas (rendering surface)
    └── View (viewport)
        ├── Scene (root node)
        │   └── Node (visual elements)
        │       ├── Image, Points, Line, Mesh, Volume, Text
        │       └── Child nodes with transforms
        └── Camera (viewing perspective)

Parent node properties (like transforms, visibility, and opacity) are composed with
child properties during rendering rather than mutating the children themselves. This
means the scene graph structure stays immutable—child property values never change,
but their effective rendered values are computed by composing ancestor properties.

Main Model Categories
---------------------
**Container Models**
    - Canvas: Top-level rendering surface (window or canvas element)
    - View: Rectangular viewport displaying a scene through a camera
    - Scene: Root container for visual elements

**Visual Nodes**
    - Image: 2D images with colormapping and intensity normalization
    - Points: Point markers with customizable symbols
    - Line: Connected polylines with per-vertex coloring
    - Mesh: Triangle mesh surfaces
    - Volume: 3D volumetric rendering
    - Text: Screen-space text labels

**Supporting Models**
    - Camera: Viewing perspective and projection
    - Transform: 4x4 affine transformations
    - ColorModel: Color specification (uniform, per-face, per-vertex)

**Interaction Models**
    - PanZoom: Pan and zoom camera interaction
    - Orbit: Orbit camera interaction
    - Letterbox: Maintain aspect ratio on view resize

Examples
--------
Build a simple scene::

    >>> from scenex.model import Scene, Image, Points
    >>> import numpy as np

    >>> scene = Scene(
    ...     children=[
    ...         Image(data=np.random.rand(100, 100)),
    ...         Points(vertices=np.random.rand(50, 2) * 100),
    ...     ]
    ... )

Create a view with an interactive camera using CanvasInteractor::

    from scenex.model import View, Camera
    from scenex.interaction import CanvasInteractor, PanZoom

    view = View(scene=scene, camera=Camera())
    canvas = snx.show(view)
    ci = CanvasInteractor(canvas)
    ci.set_controller(view, PanZoom())

Notes
-----
To display models, use `scenex.show()` which creates backend adaptors. The adaptors
handle the actual rendering by translating these declarative models into graphics
library calls (pygfx, vispy, etc.), and listen for model changes after initial setup.

See Also
--------
scenex.adaptors : Backend implementations for rendering
scenex.show : Function to display models
"""

from cmap import Color, Colormap  # re-export

from ._base import EventedBase, objects
from ._canvas import Canvas
from ._color import (
    ColorModel,
    FaceColors,
    UniformColor,
    VertexColors,
)
from ._layout import (
    Coord,
    Layout,
)
from ._nodes.camera import Camera
from ._nodes.image import Image, InterpolationMode
from ._nodes.line import Line
from ._nodes.mesh import Mesh
from ._nodes.node import AnyNode, BlendMode, Node
from ._nodes.points import Points, ScalingMode, SymbolName
from ._nodes.scene import Scene
from ._nodes.text import Text
from ._nodes.volume import RenderMode, Volume
from ._transform import Transform
from ._view import View

__all__ = [
    "AnyNode",
    "BlendMode",
    "Camera",
    "Canvas",
    "Color",
    "ColorModel",
    "Colormap",
    "Coord",
    "EventedBase",
    "FaceColors",
    "Image",
    "InterpolationMode",
    "Layout",
    "Line",
    "Mesh",
    "Node",
    "Points",
    "RenderMode",
    "ScalingMode",
    "Scene",
    "SymbolName",
    "Text",
    "Transform",
    "UniformColor",
    "VertexColors",
    "View",
    "Volume",
    "objects",
]

for obj in list(globals().values()):
    if isinstance(obj, type) and issubclass(obj, EventedBase):
        obj.__module__ = __name__
        obj.model_rebuild()
