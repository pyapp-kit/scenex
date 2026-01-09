from typing import TYPE_CHECKING, Any, Literal

from .node import Node

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Unpack

    from .node import NodeKwargs


class Scene(Node):
    """The root container node for a scene graph.

    Scene is a specialized Node that serves as the root of a scene graph hierarchy.
    It contains all the visual elements (Images, Points, Lines, Meshes, etc.) and
    cameras that make up a complete 3D scene. While functionally identical to a Node,
    Scene provides semantic clarity that this is the top-level container.

    A Scene is typically associated with a View, which pairs it with a Camera to define
    what is rendered and how. Multiple views can display the same scene from different
    camera perspectives.

    Examples
    --------
    Create a scene with visual elements:
        >>> import numpy as np
        >>> my_image = np.random.rand(100, 100).astype(np.float32)
        >>> my_points = np.random.rand(100, 3).astype(np.float32)
        >>> scene = Scene(
        ...     children=[
        ...         Image(data=my_image),
        ...         Points(
        ...             coords=my_points,
        ...             face_color=UniformColor(color=Color("red")),
        ...         ),
        ...     ]
        ... )

    Create an empty scene and later add children:
        >>> scene = Scene()
        >>> scene.add_child(Image(data=my_image))
        >>> scene.add_child(Points(coords=my_points))

    Create a hierarchical scene with nested nodes:
        >>> grandchild = Image(data=my_image)
        >>> parent = Points(coords=my_points)
        >>> scene = Scene(children=[parent])

    Use a scene with a view:
        >>> view = View(scene=scene, camera=Camera())
        >>> canvas = Canvas()
        >>> canvas.grid.add(view)

    Notes
    -----
    Scene inherits all Node attributes and methods including transform, visible,
    opacity, and children management. The scene itself does not have visual
    representation; it only serves as a container for renderable nodes.
    """

    node_type: Literal["scene"] = "scene"

    # tell mypy and pyright that this takes children, just like Node
    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            children: Iterable["Node | dict[str, Any]"] = (),
            **data: "Unpack[NodeKwargs]",
        ) -> None: ...
