from scenex import Canvas, Fractional, OffsetPlusSize, PixelGaps, View
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
    view.layout.x_span = OffsetPlusSize(offset=40, size=40)
    view.layout.y_span = OffsetPlusSize(offset=40, size=40)
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
    view.layout.x_span = OffsetPlusSize(offset=-40, size=40)
    view.layout.y_span = OffsetPlusSize(offset=-40, size=40)
    canvas = Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (60, 60, 40, 40)
    # When the canvas size changes, the view should move right
    canvas.width = 200
    assert canvas.rect_for(view) == (160, 60, 40, 40)
    canvas.height = 200
    assert canvas.rect_for(view) == (160, 160, 40, 40)


def test_span_mixture() -> None:
    """Ensures mixturing different span types on x vs y axes works correctly."""
    view = View()
    view.layout.x_span = PixelGaps(left=40, right=40)
    view.layout.y_span = OffsetPlusSize(offset=-40, size=40)
    canvas = Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (40, 60, 20, 40)
    # When the canvas size changes, the view should move right
    canvas.width = 200
    assert canvas.rect_for(view) == (40, 60, 120, 40)
    canvas.height = 200
    assert canvas.rect_for(view) == (40, 160, 120, 40)


def test_stretch_to_right_pixel_layout() -> None:
    view = View()
    # Leave 40 px gap on top and left, but also stretch to the right
    view.layout.x_span = PixelGaps(left=40, right=40)
    view.layout.y_span = PixelGaps(left=40, right=40)
    canvas = Canvas(views=[view], width=100, height=100)
    # Resulting in a 20x20 px view in the center
    assert canvas.rect_for(view) == (40, 40, 20, 20)
    # When the canvas size changes, the view should get larger
    canvas.width = 200
    assert canvas.rect_for(view) == (40, 40, 120, 20)
    canvas.height = 200
    assert canvas.rect_for(view) == (40, 40, 120, 120)


def test_fractional_region_full_canvas() -> None:
    """Default FractionalRegion (start=0, end=1, total=1) fills the entire canvas."""
    view = View()
    view.layout.x_span = Fractional(start=0, end=1, total=1)
    view.layout.y_span = Fractional(start=0, end=1, total=1)
    canvas = Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (0, 0, 100, 100)
    canvas.width = 200
    assert canvas.rect_for(view) == (0, 0, 200, 100)
    canvas.height = 50
    assert canvas.rect_for(view) == (0, 0, 200, 50)


def test_fractional_region_nonzero_start() -> None:
    """FractionalRegion with non-zero start offsets the view from the canvas edge."""
    view = View()
    # Right half horizontally, full height vertically
    view.layout.x_span = Fractional(start=1, end=2, total=2)
    canvas = Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (50, 0, 50, 100)
    canvas.width = 200
    assert canvas.rect_for(view) == (100, 0, 100, 100)


def test_fractional_region_covers_all_pixels() -> None:
    """Adjacent fractional regions together cover every pixel with no gaps or overlaps.

    With 101px / 3 parts, the final two parts absorb half of the 2-pixel remainder.
    """
    views = [View() for _ in range(3)]
    for k, v in enumerate(views):
        # Divide horizontally into thirds, full height
        v.layout.x_span = Fractional(start=k, end=k + 1, total=3)
    canvas = Canvas(views=views, width=101, height=100)
    assert canvas.rect_for(views[0]) == (0, 0, 33, 100)
    assert canvas.rect_for(views[1]) == (33, 0, 34, 100)
    assert canvas.rect_for(views[2]) == (67, 0, 34, 100)
    # Total width accounts for every pixel
    rects = [canvas.rect_for(v) for v in views]
    assert sum(r[2] for r in rects) == canvas.width


def test_layout_serialization() -> None:
    layout = Layout(x_span=Fractional(start=1, end=3, total=4))
    json = layout.model_dump_json()
    layout2 = Layout.model_validate_json(json)
    assert isinstance(layout2.x_span, Fractional)
    assert layout2.x_span.start == 1
    assert layout2.x_span.end == 3
    assert layout2.x_span.total == 4

    layout.x_span = OffsetPlusSize(offset=0, size=40)
    json = layout.model_dump_json()
    layout2 = Layout.model_validate_json(json)
    assert isinstance(layout2.x_span, OffsetPlusSize)
    assert layout2.x_span.offset == 0
    assert layout2.x_span.size == 40

    layout.x_span = PixelGaps(left=0, right=40)
    json = layout.model_dump_json()
    layout2 = Layout.model_validate_json(json)
    assert isinstance(layout2.x_span, PixelGaps)
    assert layout2.x_span.left == 0
    assert layout2.x_span.right == 40
