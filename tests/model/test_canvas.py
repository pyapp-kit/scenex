import scenex as snx


def test_multiple_views() -> None:
    # Create a canvas with two views
    view1 = snx.View()  # Left half
    view1.layout.x = "0%", "50%"
    view2 = snx.View()  # Right half
    view2.layout.x = "50%", "100%"
    canvas = snx.Canvas(views=[view1, view2])

    x1, y1, w1, h1 = canvas.rect_for(view1)
    x2, y2, w2, h2 = canvas.rect_for(view2)

    # By default the views are equally sized and side-by-side
    assert w1 == w2
    assert h1 == h2
    assert x1 + w1 == x2
    assert y1 == y2

    # Changing the canvas size should preserve the equal-split relationship
    canvas.width = canvas.width // 2
    canvas.height = canvas.height * 2

    x1, y1, w1, h1 = canvas.rect_for(view1)
    x2, y2, w2, h2 = canvas.rect_for(view2)
    assert w1 == w2
    assert h1 == h2
    assert x1 + w1 == x2
    assert y1 == y2
