import scenex as snx


# FIXME: This test probably belongs in test_view, or maybe test_layout
def test_show_canvas_size(basic_view: snx.View) -> None:
    """Tests that show_canvas respects the size of the canvas."""
    canvas = snx.show(basic_view)
    view_rect = canvas.rect_for(basic_view)
    assert canvas.width == view_rect[2]
    assert canvas.height == view_rect[3]
