import pytest

import scenex as snx


def test_show_canvas_size() -> None:
    """Ensures a (default) View passed to `show()` spans the entire canvas."""
    basic_view = snx.View()
    canvas = snx.show(basic_view)
    view_rect = canvas.rect_for(basic_view)
    assert canvas.width == view_rect[2]
    assert canvas.height == view_rect[3]


def test_native() -> None:
    """Test the native function."""
    canvas = snx.Canvas()
    # Assert an error is raised when no adaptor exists and create=False
    with pytest.raises(KeyError):
        native_widget = snx.native(canvas, create=False)
    # Assert the native widget is returned when create=True
    native_widget = snx.native(canvas, create=True)
    assert native_widget is not None
    # Assert the native widget is the same as the one returned by create=True
    native_widget2 = snx.native(canvas, create=False)
    assert native_widget is native_widget2
