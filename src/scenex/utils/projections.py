"""Utilities for creating projection matrices."""

from math import pi, tan

import pylinalg as la

from scenex.model._transform import Matrix3D, Transform


def orthographic(
    width: float,
    height: float,
) -> Transform:
    """Creates an orthographic projection matrix.

    Parameters
    ----------
    width: float
        The width of the camera rectangle
    height: float
        The height of the camera rectangle

    Returns
    -------
    projection: Transform
        A Transform matrix creating an orthographic camera view
    """
    return Transform().scaled((2 / width, 2 / height, 1))


def perspective(
    zoom_factor: float,
    fov: float,
    view_width: float,
    view_height: float,
    depth: float,
    aspect: float = 1.0,
    maintain_aspect: bool = True,
    canvas_aspect: float = 1.0,
) -> Transform:
    """Creates a perspective projection matrix.

    Derived from
    https://github.com/pygfx/pygfx/blob/e5d918c010f0de1168aefe309f9cc9279851a9b4/pygfx/cameras/_perspective.py#L348

    TODO: Explain this code. This reference may contain the information necessary:
    https://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera/virtual-pinhole-camera-model.html

    Parameters
    ----------
    zoom_factor: float
        TODO
    fov: float
        Controls how much of the scene is viewed.
    view_width: float
        TODO
    view_height: float
        TODO
        TODO: Consider passing the view model through instead of these parameters
    depth: float
        TODO
    aspect: float
        Frustum aspect radio (width / height)
    maintain_aspect: bool = True
        Whether to conform to the aspect ratio of the canvas if it differs from the
        aspect ratio of the frustum. Default True
    canvas_aspect: float
        Canvas aspect ratio
        TODO: Can't we just pass one of the two through?

    Returns
    -------
    projection: Transform
        A Transform matrix creating an orthographic camera view
    """
    matrix = Matrix3D((4, 4))

    near, far = _get_near_and_far_plane(fov, depth)

    # if self._view_offset is not None:
    #     # The view_offset should override the aspect, via its full (virtual) size
    #     view_aspect = (
    #         self._view_offset["full_width"] / self._view_offset["full_height"]
    #     )

    if fov > 0:
        # Get the reference width / height
        size = 2 * near * tan(pi / 180 * 0.5 * fov) / zoom_factor
        # Pre-apply the reference aspect ratio
        height = 2 * size / (1 + aspect)
        width = height * aspect
        # Increase either the width or height, depending on the view size
        if maintain_aspect:
            if aspect < canvas_aspect:
                width *= canvas_aspect / aspect
            else:
                height *= aspect / canvas_aspect
        # Calculate bounds
        top = +0.5 * height
        bottom = -0.5 * height
        left = -0.5 * width
        right = +0.5 * width
        # Set matrices
        projection_matrix = la.mat_perspective(
            left, right, top, bottom, near, far, depth_range=(0, 1), out=matrix
        )

    else:
        # The reference view plane is scaled with the zoom factor
        width = view_width / zoom_factor
        height = view_height / zoom_factor
        # Increase either the width or height, depending on the viewport shape
        aspect = width / height
        if maintain_aspect:
            if aspect < canvas_aspect:
                width *= canvas_aspect / aspect
            else:
                height *= aspect / canvas_aspect
        # Calculate bounds
        bottom = -0.5 * height
        top = +0.5 * height
        left = -0.5 * width
        right = +0.5 * width
        # Set matrices
        projection_matrix = la.mat_orthographic(
            left, right, top, bottom, near, far, depth_range=(0, 1), out=matrix
        )

    projection_matrix.flags.writeable = False
    return Transform(matrix)


def _get_near_and_far_plane(fov: float, depth: float) -> tuple[float, float]:
    if fov > 0:
        # Scale near plane with the fov to compensate for the fact that with very small
        # fov you're probably looking at something in the far distance.
        f = _fov_distance_factor(fov)
        # We want to be gentle with the factor for the near plane; making that value
        # small will cost a lot of bits in the depth buffer. The value for the far
        # buffer affects the precision near the camera much less.
        return depth * f / 100, depth * 10000
    else:
        # Look behind and in front in equal distance.
        # With a fov of 0, the depth precision is divided equally over the whole range.
        # So being able to look far in the distance, is *much* more costly than it is
        # for perspective projection. With a factor 100, you can zoom out until the
        # scene is just a few pixels before it disappears.
        return (-100 * depth, +100 * depth)


def _fov_distance_factor(fov: float) -> float:
    # It's important that controller and camera use the same distance calculations
    if fov > 0:
        fov_rad = fov * pi / 180
        factor = 0.5 / tan(0.5 * fov_rad)
    else:
        factor = 1.0
    return factor
