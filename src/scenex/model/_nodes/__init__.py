"""Scene graph node classes for visual elements and cameras.

This module contains all the node types that can be placed in a scene graph.
All nodes inherit from the base `Node` class, which provides common functionality
including transformations, visibility, opacity, parent-child relationships, and
event handling. Nodes form hierarchical trees where properties like transforms
propagate from parent to child.

Node Types
----------
**Visual Nodes** (renderable objects):
    - Image: 2D textured rectangles with colormapping
    - Points: Point markers with customizable symbols
    - Line: Connected polylines
    - Mesh: Triangle mesh surfaces
    - Volume: 3D volumetric data
    - Text: Screen-space text labels

**Special Nodes**:
    - Scene: Root container node for the scene graph
    - Camera: Defines viewing perspective and projection (not rendered)

Node Hierarchy
--------------
Nodes organize into parent-child hierarchies::

    Scene (root)
    ├── Image (with transform)
    ├── Node (container)
    │   ├── Points (child 1)
    │   └── Line (child 2)
    └── Camera

Node properties are composed during rendering: for example, a child's effective
transform is the composition of its own transform with all ancestor transforms.

Examples
--------
Create a simple image node::

    >>> import numpy as np
    >>> from scenex.model._nodes import Image
    >>> img = Image(data=np.random.rand(100, 100))

Create a hierarchy with transforms::

    >>> from scenex.model._nodes import Scene, Points
    >>> from scenex.model import Transform

    >>> # Parent node with transform
    >>> parent_points = Points(
    ...     coords=np.random.rand(50, 3),
    ...     transform=Transform().translated((10, 0, 0)),
    ... )
    >>> # Child node
    >>> child_img = Image(data=np.random.rand(100, 100))

    >>> # Add to scene
    >>> scene = Scene()
    >>> parent_points.parent = scene
    >>> child_img.parent = parent_points

See Also
--------
scenex.model.Node : Base class for all nodes
scenex.model.Transform : Transformation matrices
"""

from .node import Node  # noqa: I001  must be imported first to avoid circular imports
from .camera import Camera
from .image import Image
from .line import Line
from .mesh import Mesh
from .points import Points
from .scene import Scene
from .text import Text
from .volume import Volume

Node.model_rebuild()

__all__ = [
    "Camera",
    "Image",
    "Line",
    "Mesh",
    "Node",
    "Points",
    "Scene",
    "Text",
    "Volume",
]
