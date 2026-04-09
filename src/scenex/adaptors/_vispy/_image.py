from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import vispy.scene
import vispy.visuals
from vispy.app import Canvas
from vispy.gloo import gl
from vispy.gloo.context import get_current_canvas

from scenex.adaptors._base import ImageAdaptor

from ._node import Node

if TYPE_CHECKING:
    from collections.abc import Generator

    from cmap import Colormap
    from numpy.typing import ArrayLike

    from scenex import model
    from scenex.model._transform import Transform

logger = logging.getLogger("scenex.adaptors.vispy")

# Certain numpy data types are not supported by vispy. We downcast them to another type
# defined in this dict.
# See the following link for supported data types (subject to change):
# https://github.com/vispy/vispy/blob/0a6da357f091bc3966abee805ff01914105e0979/vispy/visuals/_scalable_textures.py#L370
DOWNCASTS: dict[np.dtype, np.dtype] = {
    # VisPy only supports 32-bit floats
    np.dtype("float64"): np.dtype("float32"),
    # Vispy only supports <=uint16, and no signed options
    np.dtype("int32"): np.dtype("uint16"),
    np.dtype("int64"): np.dtype("uint16"),
    np.dtype("uint32"): np.dtype("uint16"),
    np.dtype("uint64"): np.dtype("uint16"),
}


class Image(Node, ImageAdaptor):
    """pygfx backend adaptor for an Image node."""

    _vispy_node: vispy.visuals.ImageVisual

    def __init__(self, image: model.Image, **backend_kwargs: Any) -> None:
        self._model = image
        # Initialize the vispy node with dummy data
        self._vispy_node = vispy.scene.Image(
            data=np.zeros((1, 1), dtype=np.uint8),
            texture_format="auto",
            **backend_kwargs,
        )
        # Then set the data through the setter to ensure all processing is applied
        self._snx_set_data(image.data)

    def _snx_set_transform(self, arg: Transform) -> None:
        # Offset accounting for vispy's pixel centers at half-integer locations
        offset = arg.map([-0.5, -0.5, 0, 0])
        # Compensate for downscaled textures
        # Note that image axes are YX
        y_fac = self._model.data.shape[0] / self._vispy_node._data.shape[0]  # pyright: ignore
        x_fac = self._model.data.shape[1] / self._vispy_node._data.shape[1]  # pyright: ignore
        super()._snx_set_transform(arg.scaled((x_fac, y_fac, 1)).translated(offset))

    def _snx_set_cmap(self, arg: Colormap) -> None:
        self._vispy_node.cmap = arg.to_vispy()
        self._vispy_node.update()

    def _snx_set_clims(self, arg: tuple[float, float] | None) -> None:
        self._vispy_node.clim = arg
        self._vispy_node.update()

    def _snx_set_gamma(self, arg: float) -> None:
        self._vispy_node.gamma = arg
        self._vispy_node.update()

    def _snx_set_interpolation(self, arg: model.InterpolationMode) -> None:
        self._vispy_node.interpolation = arg
        self._vispy_node.update()

    def _snx_set_data(self, data: ArrayLike) -> None:
        # Coerce the data to something that can be displayed by vispy
        processed = _coerce_data(data, n_spatial=2)
        self._vispy_node.set_data(processed)

        # Update transform in case downsampling has changed the compensation factors.
        self._snx_set_transform(self._model.transform)


@contextmanager
def _opengl_context() -> Generator[None, None, None]:
    """Assure we are running with a valid OpenGL context.

    Only create a Canvas if one doesn't exist. Creating and closing a
    Canvas causes vispy to process Qt events which can cause problems.
    """
    canvas = Canvas(show=False) if get_current_canvas() is None else None
    try:
        yield
    finally:
        if canvas is not None:
            canvas.close()


@lru_cache(maxsize=1)
def _get_max_texture_sizes() -> tuple[int | None, int | None]:
    """Return the maximum texture sizes for 2D and 3D rendering.

    Returns
    -------
    Tuple[int | None, int | None]
        The max textures sizes for (2d, 3d) rendering.
    """
    with _opengl_context():
        max_size_2d = gl.glGetParameter(gl.GL_MAX_TEXTURE_SIZE)

    if not max_size_2d:
        max_size_2d = None

    # vispy/gloo doesn't provide the GL_MAX_3D_TEXTURE_SIZE location,
    # but it can be found in this list of constants
    # http://pyopengl.sourceforge.net/documentation/pydoc/OpenGL.GL.html
    with _opengl_context():
        GL_MAX_3D_TEXTURE_SIZE = 32883
        max_size_3d = gl.glGetParameter(GL_MAX_3D_TEXTURE_SIZE)

    if not max_size_3d:
        max_size_3d = None

    return max_size_2d, max_size_3d


def _downsample_data(data: np.ndarray, max_size: int, n_spatial: int) -> np.ndarray:
    """Downsample the spatial axes of *data* so none exceeds *max_size*.

    Uses a strided NumPy view (no copy). Any trailing non-spatial axis (e.g. a
    colour channel) is left untouched.

    Returns the (possibly downsampled) array.
    """
    # If the data has more than n_spatial axes, assume the last one is a channel (RGBA).
    has_channel = data.ndim > n_spatial
    strides = tuple(
        int(np.ceil(s / max_size)) if s > max_size else 1
        for s in data.shape[:n_spatial]
    )
    if all(s == 1 for s in strides):
        return data
    logger.warning(
        "Data shape %s exceeds max texture dimension (%d) and will be "
        "downsampled for rendering (strides: %s).",
        data.shape,
        max_size,
        strides,
    )
    slices: tuple[slice, ...] = tuple(slice(None, None, s) for s in strides)
    if has_channel:
        slices = (*slices, slice(None))
    return data[slices]


def _coerce_data(data: ArrayLike, n_spatial: Literal[2, 3]) -> np.ndarray:
    """Downcast and downsample *data* for GPU upload.

    Returns a (possibly strided) view — callers that need to own the
    buffer must ``.copy()`` it.
    """
    data = np.asarray(data)
    if data.dtype in DOWNCASTS:
        cast_to = DOWNCASTS[data.dtype]
        logger.warning(
            "Downcasting %s data from %s to %s for vispy compatibility",
            "image" if n_spatial == 2 else "volume",
            data.dtype.name,
            cast_to.name,
        )
        data = data.astype(cast_to)  # astype always returns a copy

    max_size = _get_max_texture_sizes()[0 if n_spatial == 2 else 1]  # 2D or 3D limit
    if max_size is not None:
        data = _downsample_data(data, max_size, n_spatial=n_spatial)

    return data
