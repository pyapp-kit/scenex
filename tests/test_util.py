import scenex as snx


def test_show_canvas_size(basic_view: snx.View) -> None:
    """Tests that show_canvas respects the size of the canvas."""
    canvas = snx.show(basic_view)
    assert canvas.width == basic_view.layout.width
    assert canvas.height == basic_view.layout.height
