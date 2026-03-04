import scenex as snx
from scenex.model._layout import ComposedDim, Fraction, Layout, Pixel, fr, px


def test_default_layout_full_canvas() -> None:
    """A default Layout fills the entire canvas."""
    view = snx.View()
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (0, 0, 100, 100)
    canvas.width = 200
    assert canvas.rect_for(view) == (0, 0, 200, 100)


def test_pixel_layout() -> None:
    """Fixed-pixel placement stays the same regardless of canvas size."""
    view = snx.View()
    view.layout.x = px(40), px(80)
    view.layout.y = px(40), px(80)
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (40, 40, 40, 40)
    canvas.width = 200
    assert canvas.rect_for(view) == (40, 40, 40, 40)
    canvas.height = 200
    assert canvas.rect_for(view) == (40, 40, 40, 40)


def test_negative_pixel_layout() -> None:
    """Negative pixel values anchor position to the far edge."""
    view = snx.View()
    # 40px from the right/bottom edge, 40px wide/tall
    view.layout.x = px(-40), fr(1)
    view.layout.y = px(-40), fr(1)
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (60, 60, 40, 40)
    canvas.width = 200
    assert canvas.rect_for(view) == (160, 60, 40, 40)
    canvas.height = 200
    assert canvas.rect_for(view) == (160, 160, 40, 40)


def test_span_mixture() -> None:
    """Different Dim expressions on x vs y axes resolve independently."""
    view = snx.View()
    # x: 40px inset on each side
    view.layout.x = px(40), px(-40)
    # y: anchored 40px wide to the far edge
    view.layout.y = px(-40), fr(1)
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (40, 60, 20, 40)
    canvas.width = 200
    assert canvas.rect_for(view) == (40, 60, 120, 40)
    canvas.height = 200
    assert canvas.rect_for(view) == (40, 160, 120, 40)


def test_stretch_layout() -> None:
    """Inset from all four edges stretches as the canvas grows."""
    view = snx.View()
    view.layout.x = px(40), px(-40)
    view.layout.y = px(40), px(-40)
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (40, 40, 20, 20)
    canvas.width = 200
    assert canvas.rect_for(view) == (40, 40, 120, 20)
    canvas.height = 200
    assert canvas.rect_for(view) == (40, 40, 120, 120)


def test_fractional_full_canvas() -> None:
    """fr(0) to fr(1) fills the entire canvas and scales with it."""
    view = snx.View()
    view.layout.x = fr(0), fr(1)
    view.layout.y = fr(0), fr(1)
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (0, 0, 100, 100)
    canvas.width = 200
    assert canvas.rect_for(view) == (0, 0, 200, 100)
    canvas.height = 50
    assert canvas.rect_for(view) == (0, 0, 200, 50)


def test_fractional_nonzero_start() -> None:
    """A fractional start offsets the view from the canvas edge."""
    view = snx.View()
    view.layout.x = fr(0.5), fr(1)
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (50, 0, 50, 100)
    canvas.width = 200
    assert canvas.rect_for(view) == (100, 0, 100, 100)


def test_fractional_covers_all_pixels() -> None:
    """Adjacent fractional views cover every pixel with no gaps or overlaps."""
    views = [snx.View() for _ in range(3)]
    for k, v in enumerate(views):
        v.layout.x = fr(k / 3), fr((k + 1) / 3)
    canvas = snx.Canvas(views=views, width=101, height=100)
    assert canvas.rect_for(views[0]) == (0, 0, 33, 100)
    assert canvas.rect_for(views[1]) == (33, 0, 34, 100)
    assert canvas.rect_for(views[2]) == (67, 0, 34, 100)
    rects = [canvas.rect_for(v) for v in views]
    assert sum(r[2] for r in rects) == canvas.width


def test_mixed_units() -> None:
    """Dim arithmetic combines fractional and pixel components correctly."""
    view = snx.View()
    # Centered 40px-wide strip
    view.layout.x = fr(0.5) - px(20), fr(0.5) + px(20)
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (30, 0, 40, 100)
    canvas.width = 200
    assert canvas.rect_for(view) == (80, 0, 40, 100)


def test_dim_arithmetic() -> None:
    """Dim operators produce correct composed values."""
    assert (fr(0.5) + px(10)) == ComposedDim(
        dim1=Fraction(num=1, denom=2), dim2=Pixel(pixels=10), operand="add"
    )
    assert (fr(1) - px(40)) == ComposedDim(
        dim1=Fraction(num=1, denom=1), dim2=Pixel(pixels=40), operand="sub"
    )
    assert (-px(20)) == Pixel(pixels=-20)
    assert (2 * fr(0.25)) == Fraction(num=1, denom=2)


def test_layout_serialization() -> None:
    """Layout round-trips through JSON with Dim fields intact."""
    layout = Layout(x_start=fr(0.25), x_end=fr(0.75))

    json_str = layout.model_dump_json()
    layout2 = Layout.model_validate_json(json_str)
    assert layout2.x_start == Fraction(num=1, denom=4)
    assert layout2.x_end == Fraction(num=3, denom=4)

    layout = Layout(x_start=fr(0.5) - px(200), x_end=fr(0.5) + px(200))
    json_str = layout.model_dump_json()
    layout2 = Layout.model_validate_json(json_str)
    assert layout2.x_start == ComposedDim(
        dim1=Fraction(num=1, denom=2), dim2=Pixel(pixels=200), operand="sub"
    )
    assert layout2.x_end == ComposedDim(
        dim1=Fraction(num=1, denom=2), dim2=Pixel(pixels=200), operand="add"
    )
