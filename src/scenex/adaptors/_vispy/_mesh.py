from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import vispy.color
import vispy.scene
import vispy.visuals

from scenex.adaptors._base import MeshAdaptor
from scenex.model._color import ColorModel, FaceColors, UniformColor, VertexColors

from ._node import Node

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from scenex import model

logger = logging.getLogger("scenex.adaptors.pygfx")


class Mesh(Node, MeshAdaptor):
    """vispy backend adaptor for an Mesh node."""

    _vispy_node: vispy.visuals.MeshVisual

    def __init__(self, mesh: model.Mesh, **backend_kwargs: Any) -> None:
        self._model = mesh
        self._colors: Any = Mesh._color_from(mesh.color)

        kwargs = {
            "vertices": np.asarray(mesh.vertices, dtype=np.float32),
            "faces": np.asarray(mesh.faces, dtype=np.uint32),
            self._colors[0]: self._colors[1],
        }

        self._vispy_node = vispy.scene.Mesh(**kwargs)

    def _snx_set_vertices(self, arg: ArrayLike) -> None:
        self._update_vispy_data()

    def _snx_set_faces(self, arg: ArrayLike) -> None:
        self._update_vispy_data()

    def _snx_set_color(self, arg: ColorModel) -> None:
        self._colors = Mesh._color_from(arg)
        self._update_vispy_data()

    def _update_vispy_data(self) -> None:
        # All of the _snx setters that deal with the "set_data" method pass through
        # here. We must remember and pass through all of these parameters every time,
        self._vispy_node.set_data(
            vertices=self._model.vertices,
            faces=np.asarray(self._model.faces, dtype=np.uint32),
            **{self._colors[0]: self._colors[1]},
        )
        self._vispy_node.update()

    @staticmethod
    def _color_from(
        arg: ColorModel,
    ) -> tuple[str, vispy.color.Color | np.ndarray]:
        if isinstance(arg, UniformColor):
            return "color", vispy.color.Color(arg.color.hex)
        elif isinstance(arg, VertexColors):
            return "vertex_colors", np.asarray(
                [c.rgba for c in arg.color], dtype=np.float32
            )
        elif isinstance(arg, FaceColors):
            # TODO: per-face colors are tricky under this paradigm.
            # vispy expects that if you provide N face colors, you have N triangles,
            # and vice-versa. This means that the intermediate state in between updating
            # the faces and the face colors will be invalid according to vispy.
            # We likely will need some strategy for batching model updates.
            logger.warning(
                "Mesh face colors are not directly supported in vispy backend."
            )

            # return "face_colors", np.asarray(
            #     [c.rgba for c in arg.color], dtype=np.float32
            # )
        raise ValueError("Invalid color model for Mesh.")
