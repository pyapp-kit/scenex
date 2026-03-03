from scenex import Canvas, FractionalRegion, PixelRegion, View
from scenex.model._layout import Layout


def test_default_layout_full_canvas() -> None:
    """Ensures a single "default" view fills the entire canvas."""
    view = View()
    canvas = Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (0, 0, 100, 100)
    # When the canvas size changes, the view should still fill the entire canvas.
    canvas.width = 200
    assert canvas.rect_for(view) == (0, 0, 200, 100)


def test_pixel_layout() -> None:
    """Ensures PixelRegion positions views correctly."""
    view = View()
    view.layout.region = PixelRegion(left=40, top=40, width=40, height=40)
    canvas = Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (40, 40, 40, 40)
    # When the canvas size changes, the view should move right
    canvas.width = 200
    assert canvas.rect_for(view) == (40, 40, 40, 40)
    canvas.height = 200
    assert canvas.rect_for(view) == (40, 40, 40, 40)


def test_negative_pixel_layout() -> None:
    """Ensures PixelRegion positions views correctly with negative coordinates."""
    view = View()
    view.layout.region = PixelRegion(left=-40, top=-40, width=40, height=40)
    canvas = Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (60, 60, 40, 40)
    # When the canvas size changes, the view should move right
    canvas.width = 200
    assert canvas.rect_for(view) == (160, 60, 40, 40)
    canvas.height = 200
    assert canvas.rect_for(view) == (160, 160, 40, 40)


def test_left_right_pixel_layout() -> None:
    """Ensures PixelRegion can be defined with left+right instead of left+width."""
    view = View()
    # Leave 40 px gap on top and left
    view.layout.region = PixelRegion(left=40, right=80, top=40, bottom=80)
    canvas = Canvas(views=[view], width=100, height=100)
    # Resulting in a 20x20 px view in the center
    assert canvas.rect_for(view) == (40, 40, 40, 40)
    # When the canvas size changes, the view should get larger
    canvas.width = 200
    assert canvas.rect_for(view) == (40, 40, 40, 40)
    canvas.height = 200
    assert canvas.rect_for(view) == (40, 40, 40, 40)


def test_stretch_to_right_pixel_layout() -> None:
    view = View()
    # Leave 40 px gap on top and left, but also stretch to the right
    view.layout.region = PixelRegion(left=40, top=40)
    canvas = Canvas(views=[view], width=100, height=100)
    # Resulting in a 20x20 px view in the center
    assert canvas.rect_for(view) == (40, 40, 60, 60)
    # When the canvas size changes, the view should get larger
    canvas.width = 200
    assert canvas.rect_for(view) == (40, 40, 160, 60)
    canvas.height = 200
    assert canvas.rect_for(view) == (40, 40, 160, 160)


def test_negative_right_bottom_pixel_layout() -> None:
    """PixelRegion with negative right/bottom counts from the canvas far edge."""
    view = View()
    view.layout.region = PixelRegion(left=0, right=-20, top=0, bottom=-20)
    canvas = Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (0, 0, 80, 80)
    # The gap tracks the far edge as the canvas resizes
    canvas.width = 200
    assert canvas.rect_for(view) == (0, 0, 180, 80)
    canvas.height = 200
    assert canvas.rect_for(view) == (0, 0, 180, 180)


def test_fractional_region_full_canvas() -> None:
    """Default FractionalRegion (start=0, end=1, total=1) fills the entire canvas."""
    view = View()
    view.layout.region = FractionalRegion()
    canvas = Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (0, 0, 100, 100)
    canvas.width = 200
    assert canvas.rect_for(view) == (0, 0, 200, 100)
    canvas.height = 50
    assert canvas.rect_for(view) == (0, 0, 200, 50)


def test_fractional_region_scalar() -> None:
    """Scalar start/end/total applies the same fraction to both x and y."""
    view = View()
    # Middle third of both dimensions
    view.layout.region = FractionalRegion(start=1, end=2, total=3)
    canvas = Canvas(views=[view], width=99, height=99)
    assert canvas.rect_for(view) == (33, 33, 33, 33)
    # Proportionally scales with canvas
    canvas.width = 198
    assert canvas.rect_for(view) == (66, 33, 66, 33)


def test_fractional_region_tuple_axes() -> None:
    """Tuple start/end/total allows independent fractions per axis."""
    view = View()
    # Left half horizontally, full height vertically
    view.layout.region = FractionalRegion(start=(0, 0), end=(1, 1), total=(2, 1))
    canvas = Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (0, 0, 50, 100)
    canvas.width = 200
    assert canvas.rect_for(view) == (0, 0, 100, 100)
    canvas.height = 50
    assert canvas.rect_for(view) == (0, 0, 100, 50)


def test_fractional_region_nonzero_start() -> None:
    """FractionalRegion with non-zero start offsets the view from the canvas edge."""
    view = View()
    # Right half horizontally, full height vertically
    view.layout.region = FractionalRegion(start=(1, 0), end=(2, 1), total=(2, 1))
    canvas = Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (50, 0, 50, 100)
    canvas.width = 200
    assert canvas.rect_for(view) == (100, 0, 100, 100)


def test_fractional_region_covers_all_pixels() -> None:
    """Adjacent fractional regions together cover every pixel with no gaps or overlaps.

    With 100px / 3 parts, the last part absorbs the 1-pixel remainder.
    """
    views = [View() for _ in range(3)]
    for k, v in enumerate(views):
        # Divide horizontally into thirds, full height
        v.layout.region = FractionalRegion(start=(k, 0), end=(k + 1, 1), total=(3, 1))
    canvas = Canvas(views=views, width=100, height=100)
    assert canvas.rect_for(views[0]) == (0, 0, 33, 100)
    assert canvas.rect_for(views[1]) == (33, 0, 33, 100)
    assert canvas.rect_for(views[2]) == (66, 0, 34, 100)
    # Total width accounts for every pixel
    rects = [canvas.rect_for(v) for v in views]
    assert sum(r[2] for r in rects) == 100


def test_layout_serialization() -> None:
    layout = Layout(region=FractionalRegion(start=1, end=3, total=4))
    json = layout.model_dump_json()
    layout2 = Layout.model_validate_json(json)
    assert isinstance(layout2.region, FractionalRegion)
    assert layout2.region.start == 1
    assert layout2.region.end == 3
    assert layout2.region.total == 4

    layout.region = PixelRegion(left=0, width=40, top=40, bottom=40)
    json = layout.model_dump_json()
    layout2 = Layout.model_validate_json(json)
    assert isinstance(layout2.region, PixelRegion)
    assert layout2.region.left == 0
    assert layout2.region.width == 40
    assert layout2.region.top == 40
    assert layout2.region.bottom == 40
