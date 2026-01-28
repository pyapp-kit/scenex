from unittest.mock import patch

import scenex as snx
import scenex.adaptors._vispy as adaptors


def test_close() -> None:
    """Ensures that the SceneCanvas' close method is called when closing the model."""
    canvas = snx.Canvas()
    vis_canvas = adaptors.adaptors.get_adaptor(canvas, create=True)
    assert isinstance(vis_canvas, adaptors.Canvas)
    with patch.object(vis_canvas._canvas, "close") as mock_close:
        canvas.close()
    mock_close.assert_called_once()


def test_multiple_views() -> None:
    # Create a canvas with two views
    view1 = snx.View()
    view2 = snx.View()
    canvas = snx.Canvas(views=[view1, view2], width=400, height=400)
    vis_canvas = adaptors.adaptors.get_adaptor(canvas, create=True)
    assert isinstance(vis_canvas, adaptors.Canvas)
    vis_view1 = adaptors.adaptors.get_adaptor(view1, create=True)
    assert isinstance(vis_view1, adaptors.View)
    vis_view2 = adaptors.adaptors.get_adaptor(view2, create=True)
    assert isinstance(vis_view2, adaptors.View)
    # Assert that by default the viewboxes are side-by-side and equally sized
    assert vis_view1._vispy_viewbox.pos == (0, 0)
    assert vis_view2._vispy_viewbox.pos == (200, 0)
    assert vis_view1._vispy_viewbox.size == (200, 400)
    assert vis_view2._vispy_viewbox.size == (200, 400)

    # Now change the canvas size and assert a change in the views
    canvas.width = 800
    canvas.height = 600
    assert vis_view1._vispy_viewbox.pos == (0, 0)
    assert vis_view2._vispy_viewbox.pos == (400, 0)
    assert vis_view1._vispy_viewbox.size == (400, 600)
    assert vis_view2._vispy_viewbox.size == (400, 600)
