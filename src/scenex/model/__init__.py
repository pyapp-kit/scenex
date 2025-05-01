"""Model objects for the scene graph.

The entire scene graph is built from these objects.  The scene graph is a
tree of nodes, where each node is a model object.  The root of the tree is
a `Scene` object, which contains all the other nodes.  Each node can have
children, which are also nodes.

To "view" a model means to create a backend adaptor for each node in the
scene graph.  The adaptor is responsible for rendering the node and
interacting with the backend.  Adaptors live in [`scenex.adaptors`][scenex].
"""

from cmap import Color, Colormap  # re-export

from ._base import EventedBase, objects
from ._canvas import Canvas
from ._layout import Layout
from ._nodes.camera import Camera, CameraType
from ._nodes.image import Image, InterpolationMode
from ._nodes.node import AnyNode, Node
from ._nodes.points import Points, ScalingMode, SymbolName
from ._nodes.scene import Scene
from ._nodes.volume import RenderMode, Volume
from ._transform import Transform
from ._view import BlendMode, View

__all__ = [
    "AnyNode",
    "BlendMode",
    "Camera",
    "CameraType",
    "Canvas",
    "Color",
    "Colormap",
    "EventedBase",
    "Image",
    "InterpolationMode",
    "Layout",
    "Node",
    "Points",
    "RenderMode",
    "ScalingMode",
    "Scene",
    "SymbolName",
    "Transform",
    "View",
    "Volume",
    "objects",
]

for obj in list(globals().values()):
    if isinstance(obj, type) and issubclass(obj, EventedBase):
        obj.__module__ = __name__
        obj.model_rebuild()
