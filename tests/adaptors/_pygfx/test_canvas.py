from unittest.mock import MagicMock

import pygfx

import scenex as snx
import scenex.adaptors._pygfx as adaptors


def test_grid() -> None:
    # Create a canvas with two views
    canvas = snx.Canvas()
    view1 = snx.View()
    view2 = snx.View()
    canvas.grid.add(view1, row=0, col=0)
    canvas.grid.add(view2, row=1, col=1)
    py_canvas = adaptors.adaptors.get_adaptor(canvas, create=True)
    assert isinstance(py_canvas, adaptors.Canvas)
    py_view1 = adaptors.adaptors.get_adaptor(view1, create=True)
    assert isinstance(py_view1, adaptors.View)
    py_view2 = adaptors.adaptors.get_adaptor(view2, create=True)
    assert isinstance(py_view2, adaptors.View)
    # Assert that by default the row/columns are equally sized
    # ...this is tough to test :)
    renderer_mock = MagicMock(spec=pygfx.renderers.WgpuRenderer)
    renderer_mock.logical_size = (canvas.width, canvas.height)
    renderer_mock.physical_size = (canvas.width, canvas.height)
    py_view1._draw(renderer_mock)
    rect1 = renderer_mock.render.call_args_list[0][1]["rect"]
    renderer_mock.reset_mock()
    py_view2._draw(renderer_mock)
    rect2 = renderer_mock.render.call_args_list[0][1]["rect"]
    renderer_mock.reset_mock()
    assert rect1[2] == rect2[2]
    assert rect1[3] == rect2[3]

    # Now change the row size and assert a change in the view heights
    canvas.grid.row_sizes = (0.7, 0.3)
    canvas.grid.col_sizes = (0.7, 0.3)
    py_view1._draw(renderer_mock)
    rect1 = renderer_mock.render.call_args_list[0][1]["rect"]
    renderer_mock.reset_mock()
    py_view2._draw(renderer_mock)
    rect2 = renderer_mock.render.call_args_list[0][1]["rect"]
    renderer_mock.reset_mock()
    assert rect1[2] == 7 * rect2[2] / 3
    assert rect1[3] == 7 * rect2[3] / 3
