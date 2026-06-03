from typing import Any

from vispy.scene.subscene import SubScene
from vispy.visuals.filters import Clipper

from scenex import model

from ._node import Node


class Scene(Node):
    _vispy_node: SubScene

    def __init__(self, scene: model.Scene, **backend_kwargs: Any) -> None:
        self._vispy_node = SubScene(**backend_kwargs)

        # Normally the clipper is set by the Viewbox when it creates a scene.
        # Here, we are creating the scene ourselves, so we have to also set
        # the clipper.
        self._vispy_node._clipper = Clipper()  # pyright: ignore

        self._vispy_node.visible = scene.visible
        self._vispy_node.order = scene.order
