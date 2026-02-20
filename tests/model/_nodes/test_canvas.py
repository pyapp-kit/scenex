import numpy as np

import scenex as snx
from scenex.app.events import Ray
from scenex.utils import projections


def test_to_world() -> None:
    """Tests Canvas.to_world"""
    # Identity projection, identity transformation
    camera = snx.Camera(
        transform=snx.Transform(),
        projection=projections.orthographic(2, 2, 2),
        interactive=True,
    )
    view = snx.View(scene=snx.Scene(children=[]), camera=camera)
    canvas = snx.Canvas(
        width=int(view.layout.width),
        height=int(view.layout.height),
        views=[view],
    )

    # Test center of canvas
    canvas_pos = (view.layout.width // 2, view.layout.height // 2)
    ray = canvas.to_world(canvas_pos)
    assert ray == Ray(origin=(0, 0, 0), direction=(0, 0, -1), source=view)

    # Test top-left corner
    canvas_pos = (0, 0)
    ray = canvas.to_world(canvas_pos)
    assert ray == Ray(origin=(-1, 1, 0), direction=(0, 0, -1), source=view)

    # Test outside the view
    canvas_pos = (view.layout.width * 2, view.layout.height * 2)
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
    canvas = snx.Canvas(
        width=int(view.layout.width),
        height=int(view.layout.height),
        views=[view],
    )

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
    canvas = snx.Canvas(
        width=int(view.layout.width),
        height=int(view.layout.height),
        views=[view],
    )

    ray = canvas.to_world((0, 0))
    assert ray == Ray(origin=(-0.5, 0.5, 0), direction=(0, 0, -1), source=view)
    camera.projection = snx.Transform()


def test_multiple_views() -> None:
    # Create a canvas with two views
    view1 = snx.View()
    view2 = snx.View()
    canvas = snx.Canvas(
        width=800,
        height=600,
        views=[view1, view2],
    )

    # Assert the by default the views are equally sized
    assert view1.layout.width == view2.layout.width
    assert view1.layout.height == view2.layout.height
    # And side-by-side (we'll add other behaviors later)
    assert view1.layout.x + view1.layout.width == view2.layout.x
    assert view1.layout.y == view2.layout.y

    # Assert changing the width and height of the canvas updates the views
    canvas.width = 400
    canvas.height = 400
    assert view1.layout.width == view2.layout.width
    assert view1.layout.height == view2.layout.height
    assert view1.layout.x + view1.layout.width == view2.layout.x
    assert view1.layout.y == view2.layout.y
