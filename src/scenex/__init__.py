"""Declarative scene graph model."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import no_type_check

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
from .model.nodes.volume import Volume
from .model.view import View

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
]


# FIXME: this is a temporary solution to deal with pyright and pydantic validation
# not yet sure what the equivalent of pydantic.mypy is for pyright
@no_type_check
def show(obj: Node | View) -> None:
    """Show the scene."""
    from .adaptors.auto import get_adaptor_registry

    adaptors = get_adaptor_registry()
    if isinstance(obj, View):
        view = obj
    elif isinstance(obj, Scene):
        view = View(scene=obj)
    elif isinstance(obj, Node):
        scene = Scene(children=[obj])
        view = View(scene=scene)

    canvas = Canvas(views=[view])

    adaptor = adaptors.get_adaptor(canvas)
    canvas.show()
    cam = adaptors.get_adaptor(view.camera)
    cam._snx_set_range(0.1)
    adaptor._draw()