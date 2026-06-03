from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pygfx

from scenex.adaptors._base import ImageAdaptor

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap
    from numpy.typing import ArrayLike

    from scenex import model

logger = logging.getLogger("scenex.adaptors.pygfx")

# Certain numpy data types are not supported by pygfx. We downthem to another type
# defined in this dict. Given the options of downcasting to uint16 (causing
# clipping/wrap around) or float32 (which can represent the full range of int32/64) we
# choose the latter. We may lose some precision for large integers, but this is likely
# less bad than downcasting
CASTS = {
    np.dtype("float64"): np.dtype("float32"),
    np.dtype("int64"): np.dtype("float32"),
    np.dtype("uint64"): np.dtype("float32"),
}


class Image(Node, ImageAdaptor):
    """pygfx backend adaptor for an Image node."""

    _pygfx_node: pygfx.Image
    _material: pygfx.ImageBasicMaterial

    def __init__(self, image: model.Image, **backend_kwargs: Any) -> None:
        self._model = image
        self._material = pygfx.ImageBasicMaterial(
            clim=image.clims,
            # This value has model render order win for coplanar objects
            depth_compare="<=",
        )
        self._pygfx_node = pygfx.Image(material=self._material)
        self._snx_set_data(image.data)

    def _snx_set_cmap(self, arg: Colormap) -> None:
        if np.asarray(self._model.data).ndim == 3:
            self._material.map = None
        else:
            self._material.map = arg.to_pygfx()

    def _snx_set_clims(self, arg: tuple[float, float] | None) -> None:
        self._material.clim = arg

    def _snx_set_gamma(self, arg: float) -> None:
        self._material.gamma = arg

    def _snx_set_interpolation(self, arg: model.InterpolationMode) -> None:
        if arg == "bicubic":
            logger.warning(
                "Bicubic interpolation not supported by pygfx - falling back to linear",
            )
            self._model.interpolation = "linear"
            return
        self._material.interpolation = arg

    def _snx_set_transform(self, arg: model.Transform) -> None:
        if not hasattr(self, "_pygfx_node"):
            return  # _snx_set_data hasn't run yet to set initial factors
        # Compensate for downscaled textures
        # NOTE that image axes are YX in model space
        # but the pygfx Texture shape will be (X, Y, 1).
        y_fac = self._model.data.shape[0] / self._texture.size[1]
        x_fac = self._model.data.shape[1] / self._texture.size[0]
        self._pygfx_node.local.matrix = (
            arg.scaled((x_fac, y_fac, 1))
            .translated((0.5 * (x_fac - 1), 0.5 * (y_fac - 1), 0))
            .root.T
        )

    def _snx_set_data(self, data: ArrayLike) -> None:
        # Coerce the data to something that can be displayed by pygfx
        processed = _coerce_data(np.asanyarray(data), n_spatial=2)

        # If we have a texture already, see whether we can reuse it:
        current: pygfx.Texture | None = getattr(self, "_texture", None)
        if current is not None and _can_reuse(current, processed):
            # To reuse, we overwrite its buffer and mark dirty.
            current.data[:] = processed  # pyright: ignore
            current.update_full()
        else:
            # Copy so later in-place reuse won't mutate the originally-passed array.
            new_buffer = processed.copy()
            self._texture = pygfx.Texture(new_buffer, dim=_texture_dim(processed))
            self._pygfx_node.geometry = pygfx.Geometry(grid=self._texture)
            # Only update the material map when the array dimensionality changes
            # (grayscale ↔ RGB/RGBA) or on first creation; otherwise the existing
            # map is still correct and recreating it is wasteful.
            prev_ndim = (
                current.data.ndim
                if current is not None and current.data is not None
                else None
            )
            if prev_ndim != processed.ndim:
                self._material.map = (
                    None if processed.ndim == 3 else self._model.cmap.to_pygfx()
                )

        # Keep transform compensation in sync whenever data or factors change.
        self._snx_set_transform(self._model.transform)


def _texture_dim(data: np.ndarray) -> int:
    """Spatial dimensionality of *image data* for a pygfx Texture."""
    if data.ndim > 2 and data.shape[-1] in (3, 4):
        # RGB or RGBA image - treat as 2D with a channel dimension, not 3D
        return 2
    # Otherwise the number of spatial dimensions is just the number of dimensions
    return data.ndim


@lru_cache(maxsize=1)
def _get_max_texture_sizes() -> tuple[int | None, int | None]:
    """Return (max_2d, max_3d) texture dimensions from the wgpu adapter."""
    try:
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync()
        limits = adapter.limits
        return (
            limits.get("max-texture-dimension-2d"),
            limits.get("max-texture-dimension-3d"),
        )
    except Exception:
        return None, None


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


def _coerce_data(data: np.ndarray, n_spatial: Literal[2, 3]) -> np.ndarray:
    """Downcast and downsample *data* for GPU upload.

    Returns a (possibly strided) view — callers that need to own the
    buffer must ``.copy()`` it.
    """
    if data.dtype in CASTS:
        cast_to = CASTS[data.dtype]
        logger.warning(
            "Casting %s data from %s to %s for pygfx compatibility",
            "image" if n_spatial == 2 else "volume",
            data.dtype.name,
            cast_to.name,
        )
        data = data.astype(cast_to)  # astype always returns a copy

    max_size = _get_max_texture_sizes()[0 if n_spatial == 2 else 1]  # 2D or 3D limit
    if max_size is not None:
        data = _downsample_data(data, max_size, n_spatial=n_spatial)

    return data


def _can_reuse(texture: pygfx.Texture, processed: np.ndarray) -> bool:
    """True if *texture* can hold *processed* without reallocation."""
    return (
        texture.data is not None
        and texture.data.shape == processed.shape
        and texture.data.dtype == processed.dtype
        and texture.dim == _texture_dim(processed)
    )
