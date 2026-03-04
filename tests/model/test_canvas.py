import numpy as np

import scenex as snx
from scenex.app.events import Ray
from scenex.model._layout import fr
from scenex.utils import projections

# Default canvas size used across tests
_W, _H = 500, 500


def test_to_world() -> None:
    """Tests Canvas.to_world"""
    # Identity projection, identity transformation
    camera = snx.Camera(
        transform=snx.Transform(),
        projection=projections.orthographic(2, 2, 2),
        interactive=True,
    )
    view = snx.View(scene=snx.Scene(children=[]), camera=camera)
    canvas = snx.Canvas(width=_W, height=_H, views=[view])
    w, h = canvas.rect_for(view)[2:]

    # Test center of canvas
    canvas_pos = (w // 2, h // 2)
    ray = canvas.to_world(canvas_pos)
    assert ray == Ray(origin=(0, 0, 0), direction=(0, 0, -1), source=view)

    # Test top-left corner
    canvas_pos = (0, 0)
    ray = canvas.to_world(canvas_pos)
    assert ray == Ray(origin=(-1, 1, 0), direction=(0, 0, -1), source=view)

    # Test outside the view
    canvas_pos = (w * 2, h * 2)
    ray = canvas.to_world(canvas_pos)
    assert ray is None


def test_to_world_translated() -> None:
    """Tests Canvas.to_world with a translated camera"""
    # Identity projection, small transformation
    camera = snx.Camera(
        transform=snx.Transform().translated((1, 1, 1)),
        projection=projections.orthographic(2, 2, 2),
        interactive=True,
    )
    view = snx.View(scene=snx.Scene(children=[]), camera=camera)
    canvas = snx.Canvas(width=_W, height=_H, views=[view])

    ray = canvas.to_world((0, 0))
    assert ray == Ray(origin=(0, 2, 1), direction=(0, 0, -1), source=view)
    # Rotate counter-clockwise around +Z - we see a clockwise rotation
    # i.e. (-1, 1, 0) moves to the top right corner and (-1, -1, 0) moves to the
    # top left corner
    camera.transform = snx.Transform().rotated(90, (0, 0, 1))
    ray = canvas.to_world((0, 0))
    # Rounding errors :(
    assert ray is not None
    assert np.allclose(ray.origin, (-1, -1, 0), atol=1e-7)
    assert np.array_equal(ray.direction, (0, 0, -1))
    assert ray.source == view
    camera.transform = snx.Transform()


def test_to_world_projection() -> None:
    """Tests Canvas.to_world with a non-identity camera projection"""
    # Narrowed projection, identity transformation
    camera = snx.Camera(
        transform=snx.Transform(),
        projection=projections.orthographic(1, 1, 1),
        interactive=True,
    )
    view = snx.View(scene=snx.Scene(children=[]), camera=camera)
    canvas = snx.Canvas(width=_W, height=_H, views=[view])

    ray = canvas.to_world((0, 0))
    assert ray == Ray(origin=(-0.5, 0.5, 0), direction=(0, 0, -1), source=view)
    camera.projection = snx.Transform()


def test_multiple_views() -> None:
    # Create a canvas with two views
    view1 = snx.View()  # Left half
    view1.layout.x_start = fr(0)
    view1.layout.x_end = fr(0.5)
    view2 = snx.View()  # Right half
    view2.layout.x_start = fr(0.5)
    view2.layout.x_end = fr(1)
    canvas = snx.Canvas(width=800, height=600, views=[view1, view2])

    x1, y1, w1, h1 = canvas.rect_for(view1)
    x2, y2, w2, h2 = canvas.rect_for(view2)

    # By default the views are equally sized and side-by-side
    assert w1 == w2
    assert h1 == h2
    assert x1 + w1 == x2
    assert y1 == y2

    # Changing the canvas size should preserve the equal-split relationship
    canvas.width = 400
    canvas.height = 400

    x1, y1, w1, h1 = canvas.rect_for(view1)
    x2, y2, w2, h2 = canvas.rect_for(view2)
    assert w1 == w2
    assert h1 == h2
    assert x1 + w1 == x2
    assert y1 == y2
