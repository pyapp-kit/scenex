from typing import Any

import pygfx

from scenex import model

from ._node import Node


class Scene(Node):
    _pygfx_node: pygfx.Scene

    def __init__(self, scene: model.Scene, **backend_kwargs: Any) -> None:
        self._pygfx_node = pygfx.Scene(visible=scene.visible, **backend_kwargs)
        self._pygfx_node.render_order = scene.order
