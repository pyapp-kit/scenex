from typing import Any

from vispy.scene.subscene import SubScene

from scenex import model

from ._node import Node


class Scene(Node):
    _vispy_node: SubScene

    def __init__(self, scene: model.Scene, **backend_kwargs: Any) -> None:
        self._vispy_node = SubScene(**backend_kwargs)

        self._vispy_node.visible = scene.visible
        self._vispy_node.order = scene.order
