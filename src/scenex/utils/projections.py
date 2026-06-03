"""Utilities for creating projection matrices."""

from __future__ import annotations

from math import pi, tan
from typing import TYPE_CHECKING, Literal

import numpy as np

from scenex.model._transform import Transform

if TYPE_CHECKING:
    from scenex.model import View


def orthographic(width: float = 1, height: float = 1, depth: float = 1) -> Transform:
    """Creates an orthographic projection matrix.

    Note that the resulting projection matrix provides no positional offset; this would
    be out of scope, as such is the job of a camera's transform parameter.

    TODO: Consider passing bounds (i.e. a tuple[float, float]) for each parameter.
    Unfortunately, though, this would effectively allow positional offsets for width and
    height.

    Parameters
    ----------
    width: float, optional
        The width of the camera rectangular prism. Must be positive. Default 1
        (mirroring the side length of a unit cube).
    height: float, optional
        The height of the camera rectangular prism. Must be positive. Default 1
        (mirroring the side length of a unit cube).
    depth: float, optional
        The depth of the camera rectangular prism. Must be positive. The near and far
        clipping planes of the resulting matrix become (-depth / 2) and (depth / 2)
        respectively. Default 1, increase (to render things farther away) or decrease
        (to increase performance) as needed.

        TODO: Is this a good default? May want to consider some large number (1000?)
        instead

    Returns
    -------
    projection: Transform
        A Transform matrix creating an orthographic camera view
    """
    if any(arg <= 0 for arg in (width, height, depth)):
        # Negative values would flip the view, an unlikely user intention.
        # But it could be allowed if there's a good reason...
        raise ValueError("Orthographic projection parameters must be positive.")
    # REALLY small values could cause overflow in division, so we clamp to a min value.
    width = max(width, 1e-200)
    height = max(height, 1e-200)
    depth = max(depth, 1e-200)
    return Transform().scaled((2 / width, 2 / height, -2 / depth))


def perspective(fov: float, near: float, far: float) -> Transform:
    """Creates a perspective projection matrix.

    Note that the resulting projection matrix provides no positional offset; this would
    be out of scope, as such is the job of a camera's transform parameter.

    Parameters
    ----------
    fov: float
        The field of view of the camera rectangle.
    near: float
        The distance from the camera to the near clipping plane.
    far: float
        The distance from the camera to the far clipping plane.

    Returns
    -------
    projection: Transform
        A Transform matrix creating a perspective camera view
    """
    if fov == 0:
        raise ValueError(
            "Perspective matrices require fov>0. Maybe consider an orthographic matrix?"
        )

    matrix = np.zeros((4, 4))

    # Computations derived from
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix.html
    scaling_factor = 1 / (tan(fov / 2 * pi / 180))
    matrix[0, 0] = scaling_factor
    matrix[1, 1] = scaling_factor

    z_scale = -1 * far / (far - near)
    matrix[2, 2] = z_scale
    z_translation = -1 * far * near / (far - near)
    matrix[2, 3] = z_translation

    matrix[3, 2] = -1
    return Transform(root=matrix)


def zoom_to_fit(
    view: View,
    type: Literal["perspective", "orthographic"] = "orthographic",
    zoom_factor: float = 1.0,
    letterbox: bool = False,
) -> None:
    """Adjusts the Camera to fit the entire scene.

    Parameters
    ----------
    view: View
        The view to adjust. Contains the camera, whose parameters will be adjusted, and
        the scene, whose elements will be considered in the adjustment.
    type: Literal["perspective", "orthographic"]
        The type of canvas projection to use. Orthographic by default.
    zoom_factor: float
        The amount to zoom the scene after adjusting camera parameters. The default,
        1.0, will leave the scene touching the edges of the view. As the zoom factor
        approaches 0, the scene will linearly decrease in size. As the zoom factor
        increases beyond 1.0, the bounds of the scene will expand linearly beyond the
        view.
    letterbox: bool
        Whether to letterbox/pillarbox to prevent anisotropic distortion. When True,
        squares will appear as squares regardless of view dimensions. When False,
        content may be stretched to fill the view. Default False.
    """
    bb = view.scene.bounding_box
    center = np.mean(bb, axis=0) if bb else (0, 0, 0)
    # Note that the np.maximum avoids bounding boxes with zero width, height, or depth.
    # These wouldn't really be boxes, and would cause division by zero errors in
    # projection matrix calculations
    w, h, d = np.maximum(np.ptp(bb, axis=0) if bb else (1, 1, 1), 1e-6)

    if type == "orthographic":
        if letterbox:
            # The scene has aspect w:h. If that doesn't match the viewport's aspect
            # ratio ar, setting the orthographic frustum to those values will distort
            # world-space squares. We can correct this distortion by expanding whichever
            # dimension is too small to make w:h == ar. This is kinda like letterboxing,
            # except you'll see extra scene background instead of black bars.
            if (ar := _aspect_ratio(view)) is not None:
                if w / h > ar:  # scene wider than view - expand height
                    h = w / ar
                else:  # scene taller than view - expand width
                    w = h * ar
        view.camera.transform = Transform().translated(center)
        view.camera.projection = orthographic(w, h, d).scaled([zoom_factor] * 3)
    elif type == "perspective":
        fov = 70
        # First, we need to figure out how far away to place the camera so that the
        # entire scene fits within the frustum defined by the FOV. Calculation borrowed
        # from
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix.html
        if letterbox and (ar := _aspect_ratio(view)) is not None:
            # perspective() produces a square frustum (equal x/y FOV). If the viewport
            # isn't square, world-space squares will appear distorted. We correct this
            # by adjusting two different camera parameters. First, if the scene is wider
            # than the viewport, our distance will actually have to increased based on
            # that view aspect ratio.
            o = max(h, w / ar) / 2
        else:
            o = max(w, h) / 2
        a = o / tan(fov * pi / 360) / zoom_factor

        # Place the camera so the bounding box's front face (z = center[2] + d/2)
        # maps to the near plane of the frustum.
        z_bound = center[2] + (d / 2) + a
        view.camera.transform = Transform().translated((center[0], center[1], z_bound))
        proj = perspective(fov, near=1, far=1_000_000)
        if letterbox and (ar := _aspect_ratio(view)) is not None:
            # Second, if the viewport is non-square, we'll have to adjust our (square)
            # projection matrix to reflect the non-square viewport. The result is kinda
            # like letterboxing, except you'll see extra scene background instead of
            # black bars.
            proj = proj.scaled((1.0 / ar, 1.0, 1.0))
        view.camera.projection = proj
    else:
        raise TypeError(f"Unrecognized projection type: {type}")


def _aspect_ratio(view: View) -> float | None:
    if not (canvas := view._canvas):
        # If the view isn't attached to a canvas, we can't get viewport dimensions, and
        # can't compute an aspect ratio.
        return None
    _, _, pw, ph = canvas.rect_for(view)
    if pw <= 0 or ph <= 0:
        # If the view has non-positive dimensions, we can't compute an aspect ratio.
        return None
    return pw / ph
