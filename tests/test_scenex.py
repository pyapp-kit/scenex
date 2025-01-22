from scenex import model


def test_schema() -> None:
    assert model.Canvas.model_json_schema(mode="serialization")
    assert model.Canvas.model_json_schema(mode="validation")
