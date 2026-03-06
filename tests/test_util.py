import scenex as snx


def test_show_canvas_size() -> None:
    """Ensures a (default) View passed to `show()` spans the entire canvas."""
    basic_view = snx.View()
    canvas = snx.show(basic_view)
    view_rect = canvas.rect_for(basic_view)
    assert canvas.width == view_rect[2]
    assert canvas.height == view_rect[3]
