"""Utilities for creating projection matrices."""

from __future__ import annotations

from math import pi, tan
from typing import TYPE_CHECKING, Literal

import numpy as np

# FIXME the top import (commented out) does not work due to a circular import
# Because the Camera model uses orthographic for its default transform.
# Might want to think about a better organization.
# from scenex.model import Transform
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
        The width of the camera rectangular prism. Default 1 (mirroring the side length
        of a unit cube).
    height: float, optional
        The height of the camera rectangular prism. Default 1 (mirroring the side length
        of a unit cube).
    depth: float, optional
        The depth of the camera rectangular prism. The near and far clipping planes of
        the resulting matrix become (-depth / 2) and (depth / 2) respectively. Default
        1, increase (to render things farther away) or decrease (to increase
        performance) as needed.

        TODO: Is this a good default? May want to consider some large number (1000?)
        instead

    Returns
    -------
    projection: Transform
        A Transform matrix creating an orthographic camera view
    """
    width = width if width else 1e-6
    height = height if height else 1e-6
    depth = depth if depth else 1e-6
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


# TODO: perspective mode?
# TODO: Preserve some camera state?
def zoom_to_fit(
    view: View,
    type: Literal["perspective", "orthographic"] = "orthographic",
    zoom_factor: float = 1.0,
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
    """
    # TODO: Test whether the camera can affect the bounding box...

    bb = view.scene.bounding_box
    center = np.mean(bb, axis=0) if bb else (0, 0, 0)
    w, h, d = np.ptp(bb, axis=0) if bb else (1, 1, 1)
    if type == "orthographic":
        view.camera.transform = Transform().translated(center)
        view.camera.projection = orthographic(w, h, d).scaled([zoom_factor] * 3)
    elif type == "perspective":
        # Compute the distance a to the near plane of the frustum using a default fov
        o = max(w, h) / 2
        fov = 70
        a = o / tan(fov * pi / 360) / zoom_factor

        # So that the bounding cube's front plane is mapped to the canvas,
        # the camera must be a units away from the front plane (at z=(center[2] + d/2))
        z_bound = center[2] + (d / 2) + a
        view.camera.transform = Transform().translated((center[0], center[1], z_bound))
        # Note that the near and far planes are set to reasonable defaults.
        # TODO: Consider making these parameters
        view.camera.projection = perspective(fov, near=1, far=1_000_000)
    else:
        raise TypeError(f"Unrecognized projection type: {type}")
