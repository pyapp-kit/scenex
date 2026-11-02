from unittest.mock import patch

import pytest

import scenex as snx
from scenex.adaptors import get_adaptor_registry


@pytest.mark.parametrize("clear_all", [False, True])
def test_clear_disconnects_adaptors(clear_all: bool) -> None:
    registry = get_adaptor_registry()
    camera = snx.Camera()
    other = snx.Camera()
    adaptor = registry.get_adaptor(camera)
    other_adaptor = registry.get_adaptor(other)

    with (
        patch.object(adaptor, "_snx_set_projection") as update,
        patch.object(other_adaptor, "_snx_set_projection") as other_update,
    ):
        # Confirm the model is connected before testing disconnection.
        camera.projection = snx.Transform().scaled((2, 2, 2))
        update.assert_called_once()
        update.reset_mock()

        registry.clear(None if clear_all else camera)
        # Clearing an already removed adaptor is harmless.
        registry.clear(camera)
        camera.projection = snx.Transform()
        other.projection = snx.Transform().scaled((3, 3, 3))
        update.assert_not_called()
        if clear_all:
            other_update.assert_not_called()
            assert not tuple(registry.all())
        else:
            other_update.assert_called_once()
            assert tuple(registry.all()) == (other_adaptor,)

    with pytest.raises(KeyError):
        registry.get_adaptor(camera, create=False)


def test_clear_registry_releases_adaptors() -> None:
    registry = get_adaptor_registry()
    canvas = snx.Canvas(views=[snx.View()])
    registry.get_adaptor(canvas)
    canvas.close()
    registry.clear()
    assert not tuple(registry.all())


def test_clear_disconnects_child_events() -> None:
    registry = get_adaptor_registry()
    scene = snx.Scene()
    adaptor_class = registry.get_adaptor_class(scene)
    with (
        patch.object(adaptor_class, "_snx_add_child") as add,
        patch.object(adaptor_class, "_snx_remove_child") as remove,
    ):
        registry.get_adaptor(scene)
        child = snx.Scene()
        scene.add_child(child)
        scene.remove_child(child)
        add.assert_called_once_with(child)
        remove.assert_called_once_with(child)
        add.reset_mock()
        remove.reset_mock()

        registry.clear(scene)
        scene.add_child(child)
        scene.remove_child(child)
        add.assert_not_called()
        remove.assert_not_called()
