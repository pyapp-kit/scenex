import scenex as snx


def test_basic_view(basic_view: snx.View) -> None:
    snx.show(basic_view)
    assert isinstance(basic_view.model_dump(), dict)
    assert isinstance(basic_view.model_dump_json(), str)
