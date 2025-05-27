from unittest.mock import Mock

import scenex as snx
from scenex.adaptors import Adaptor


# recursively collect all subclasses of a type
def collect_adaptors(cls: type) -> list[type]:
    """Recursively collect all subclasses of Adaptor."""
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(collect_adaptors(subclass))
    return subclasses


ADAPTORS = collect_adaptors(Adaptor)


def test_schema() -> None:
    assert snx.Canvas.model_json_schema(mode="serialization")
    assert snx.Canvas.model_json_schema(mode="validation")


def test_changing_parent_pure_model() -> None:
    """Test that changing the parent of a model object works, and emits events."""
    # create a scene and a view
    scene1 = snx.Scene()
    scene2 = snx.Scene()
    img1 = snx.Image()
    img2 = snx.Image()

    # nothing is in any scene yet
    assert not any(x in y for x in (img1, img2) for y in (scene1, scene2))
    assert img1.parent is None
    assert img2.parent is None
    assert not scene1.children
    assert not scene2.children

    # create mock objects to test events
    mock1 = Mock()
    mock1_all = Mock()
    img1.events.parent.connect(mock1)
    img1.events.connect(mock1_all)
    mock2 = Mock()
    img2.events.parent.connect(mock2)

    # set img1's parent to scene1
    img1.parent = scene1
    # this should add img1 to scene1's children
    assert img1.parent is scene1
    assert img1 in scene1.children
    # and it should have emitted a parent event
    mock1.assert_called_once_with(scene1, None)  # old, new
    mock1_all.assert_called_once()
    emit_info = mock1_all.call_args[0][0]
    assert emit_info.signal.name == "parent"
    assert emit_info.args == (scene1, None)

    # set img2's parent to scene2, using add_child
    scene2.add_child(img2)
    # this should add img2 to scene2's children
    assert img2.parent is scene2
    assert img2 in scene2.children
    # and it should have emitted a parent event
    mock2.assert_called_once_with(scene2, None)  # old, new

    # neither image should be in the other scene
    assert img1 not in scene2.children
    assert img2 not in scene1.children

    # move img1 to scene2
    mock1.reset_mock()
    scene2.add_child(img1)
    # this should remove img1 from scene1's children and add it to scene2's
    assert img1.parent is scene2
    assert img1 in scene2.children
    assert img1 not in scene1.children
    assert not scene1.children
    # and it should have emitted a parent event
    mock1.assert_called_once_with(scene2, scene1)  # old, new
    emit_info = mock1_all.call_args[0][0]
    assert emit_info.signal.name == "parent"
    assert emit_info.args == (scene2, scene1)

    # set img2's parent to None
    mock2.reset_mock()
    img2.parent = None
    # this should remove img2 from scene2's children
    assert img2.parent is None
    assert img2 not in scene2.children
    # and it should have emitted a parent event
    mock2.assert_called_once_with(None, scene2)  # old, new
