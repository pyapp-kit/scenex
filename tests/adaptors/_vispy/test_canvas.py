import scenex as snx
import scenex.adaptors._vispy as adaptors


def test_grid() -> None:
    # Create a canvas with two views
    canvas = snx.Canvas()
    view1 = snx.View()
    view2 = snx.View()
    canvas.grid.add(view1, row=0, col=0)
    canvas.grid.add(view2, row=1, col=1)
    vis_canvas = adaptors.adaptors.get_adaptor(canvas, create=True)
    assert isinstance(vis_canvas, adaptors.Canvas)
    vis_view1 = adaptors.adaptors.get_adaptor(view1, create=True)
    assert isinstance(vis_view1, adaptors.View)
    vis_view2 = adaptors.adaptors.get_adaptor(view2, create=True)
    assert isinstance(vis_view2, adaptors.View)
    # Assert that by default the row/columns are equally sized
    assert vis_view1._vispy_viewbox.rect == vis_view2._vispy_viewbox.rect

    # Now change the row size and assert a change in the view heights
    canvas.grid.row_sizes = (0.7, 0.3)
    canvas.grid.col_sizes = (0.7, 0.3)
    # Assert that by default the row/columns are equally sized
    assert vis_view1._vispy_viewbox.width == vis_view2._vispy_viewbox.width * 7 / 3
    assert vis_view1._vispy_viewbox.height == vis_view2._vispy_viewbox.height * 7 / 3
