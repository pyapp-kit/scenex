import pytest

import scenex as snx
from scenex.model._layout import Layout, resolve_dim


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
    view.layout.x = "40px", "80px"
    view.layout.y = "40px", "80px"
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
    view.layout.x = "-40px", "100%"
    view.layout.y = "-40px", "100%"
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (60, 60, 40, 40)
    canvas.width = 200
    assert canvas.rect_for(view) == (160, 60, 40, 40)
    canvas.height = 200
    assert canvas.rect_for(view) == (160, 160, 40, 40)


def test_span_mixture() -> None:
    """Different dimension expressions on x vs y axes resolve independently."""
    view = snx.View()
    # x: 40px inset on each side
    view.layout.x = "40px", "-40px"
    # y: anchored 40px wide to the far edge
    view.layout.y = "-40px", "100%"
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (40, 60, 20, 40)
    canvas.width = 200
    assert canvas.rect_for(view) == (40, 60, 120, 40)
    canvas.height = 200
    assert canvas.rect_for(view) == (40, 160, 120, 40)


def test_stretch_layout() -> None:
    """Inset from all four edges stretches as the canvas grows."""
    view = snx.View()
    view.layout.x = "40px", "-40px"
    view.layout.y = "40px", "-40px"
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (40, 40, 20, 20)
    canvas.width = 200
    assert canvas.rect_for(view) == (40, 40, 120, 20)
    canvas.height = 200
    assert canvas.rect_for(view) == (40, 40, 120, 120)


def test_fractional_full_canvas() -> None:
    """0% to 100% fills the entire canvas and scales with it."""
    view = snx.View()
    view.layout.x = "0%", "100%"
    view.layout.y = "0%", "100%"
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (0, 0, 100, 100)
    canvas.width = 200
    assert canvas.rect_for(view) == (0, 0, 200, 100)
    canvas.height = 50
    assert canvas.rect_for(view) == (0, 0, 200, 50)


def test_fractional_nonzero_start() -> None:
    """A fractional start offsets the view from the canvas edge."""
    view = snx.View()
    view.layout.x = "50%", "100%"
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (50, 0, 50, 100)
    canvas.width = 200
    assert canvas.rect_for(view) == (100, 0, 100, 100)


def test_fractional_covers_all_pixels() -> None:
    """Adjacent fractional views cover every pixel with no gaps or overlaps."""
    views = [snx.View() for _ in range(3)]
    stops = [f"{k / 3 * 100}%" for k in range(4)]
    for k, v in enumerate(views):
        v.layout.x = stops[k], stops[k + 1]
    canvas = snx.Canvas(views=views, width=101, height=100)
    assert canvas.rect_for(views[0]) == (0, 0, 33, 100)
    assert canvas.rect_for(views[1]) == (33, 0, 34, 100)
    assert canvas.rect_for(views[2]) == (67, 0, 34, 100)
    rects = [canvas.rect_for(v) for v in views]
    assert sum(r[2] for r in rects) == canvas.width


def test_layout_serialization() -> None:
    """Layout round-trips through JSON with string Unit fields intact."""
    layout = Layout(x_start="25%", x_end="75%")
    json_str = layout.model_dump_json()
    layout2 = Layout.model_validate_json(json_str)
    assert layout2.x_start == "25%"
    assert layout2.x_end == "75%"

    layout = Layout(x_start="50px", x_end="-50px")
    json_str = layout.model_dump_json()
    layout2 = Layout.model_validate_json(json_str)
    assert layout2.x_start == "50px"
    assert layout2.x_end == "-50px"

    layout = Layout(x_start="50px + 50%", x_end="-50px + 50%")
    json_str = layout.model_dump_json()
    layout2 = Layout.model_validate_json(json_str)
    assert layout2.x_start == "50px + 50%"
    assert layout2.x_end == "-50px + 50%"


def test_mixed_units() -> None:
    """Compound expressions mixing percentages and pixels resolve correctly."""
    view = snx.View()
    # Centered 40px-wide strip: "50% - 20px" to "50% + 20px"
    view.layout.x = "50% - 20px", "50% + 20px"
    canvas = snx.Canvas(views=[view], width=100, height=100)
    assert canvas.rect_for(view) == (30, 0, 40, 100)
    canvas.width = 200
    assert canvas.rect_for(view) == (80, 0, 40, 100)


@pytest.mark.parametrize(
    ("value", "total", "expected"),
    [
        ("0%", 100, 0),
        ("33.33333333333333%", 101, 33),
        ("100%", 200, 200),
        ("-20%", 200, 160),
        ("0px", 100, 0),
        ("40px", 100, 40),
        ("-40px", 100, 60),
        ("50% + 10px", 100, 60),
        ("50% - 10px", 200, 90),
        ("10px + 50%", 100, 60),
        ("-10px + 50%", 200, 90),
        ("- 10px + 50%", 200, 90),
        ("-40px + 10px + 10px", 200, 180),
    ],
)
def test_resolve_dim(value: str, total: int, expected: int) -> None:
    assert resolve_dim(value, total) == expected


@pytest.mark.parametrize("bad", ["50", "50em", "abc%", "12.5.5px", ""])
def test_invalid_dim_rejected(bad: str) -> None:
    with pytest.raises((ValueError, Exception)):
        Layout(x_start=bad)
