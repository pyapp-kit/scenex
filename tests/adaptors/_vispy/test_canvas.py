from unittest.mock import patch

import scenex as snx
import scenex.adaptors._vispy as adaptors


def test_close() -> None:
    """Ensures that the SceneCanvas' close method is called when closing the model."""
    canvas = snx.Canvas()
    vis_canvas = adaptors.adaptors.get_adaptor(canvas, create=True)
    assert isinstance(vis_canvas, adaptors.Canvas)
    assert vis_canvas._filter is not None
    with patch.object(
        vis_canvas._filter, "uninstall", wraps=vis_canvas._filter.uninstall
    ) as uninstall:
        # Closing the model closes the native canvas and releases its adaptor.
        canvas.close()
        assert vis_canvas._canvas._closed
        assert vis_canvas._filter is None
        assert canvas._model_id.hex not in adaptors.adaptors._objects

        # Repeated closes through either entry point are harmless.
        canvas.close()
        vis_canvas._snx_close()
        uninstall.assert_called_once()


def test_multiple_views() -> None:
    # Create a canvas with two views
    view1 = snx.View()  # Left half
    view1.layout.x = "0%", "50%"
    view2 = snx.View()  # Right half
    view2.layout.x = "50%", "100%"
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
