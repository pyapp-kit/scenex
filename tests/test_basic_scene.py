import scenex as snx


def test_basic_view(basic_view: snx.View) -> None:
    snx.show(basic_view)
    assert isinstance(basic_view.model_dump(), dict)
    assert isinstance(basic_view.model_dump_json(), str)


def test_adding_node_to_existing_Scene(
    random_image_node: snx.Image, sine_image_node: snx.Image, qapp
) -> None:
    view = snx.View()

    snx.show(view)
    sine_image_node.parent = view.scene
    from rendercanvas.auto import loop

    loop.run()
    qapp.processEvents()
    x = view.canvas.render()
    qapp.processEvents()