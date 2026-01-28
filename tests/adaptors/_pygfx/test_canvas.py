from unittest.mock import MagicMock, patch

import pygfx

import scenex as snx
import scenex.adaptors._pygfx as adaptors


def test_close() -> None:
    """Ensures that the RenderCanvas is closed (soon) after closing the model."""
    canvas = snx.Canvas()
    py_canvas = adaptors.adaptors.get_adaptor(canvas, create=True)
    assert isinstance(py_canvas, adaptors.Canvas)
    with patch.object(py_canvas._wgpu_canvas, "close") as mock_close:
        canvas.close()
    mock_close.assert_called_once()


def test_multiple_views() -> None:
    # Create a canvas with two views
    view1 = snx.View()
    view2 = snx.View()
    canvas = snx.Canvas(views=[view1, view2], width=400, height=400)
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
    # Check that the views are side-by-side and equally sized
    assert rect1 == (0, 0, 200, 400)
    assert rect2 == (200, 0, 200, 400)

    # Now change the canvas size and ensure the views update accordingly
    canvas.width = 800
    canvas.height = 600
    renderer_mock = MagicMock(spec=pygfx.renderers.WgpuRenderer)
    renderer_mock.logical_size = (canvas.width, canvas.height)
    renderer_mock.physical_size = (canvas.width, canvas.height)
    py_view1._draw(renderer_mock)
    rect1 = renderer_mock.render.call_args_list[0][1]["rect"]
    renderer_mock.reset_mock()
    py_view2._draw(renderer_mock)
    rect2 = renderer_mock.render.call_args_list[0][1]["rect"]
    renderer_mock.reset_mock()
    # Check that the views are side-by-side and equally sized
    assert rect1 == (0, 0, 400, 600)
    assert rect2 == (400, 0, 400, 600)
