import numpy as np

import scenex as snx


def test_zoom_to_fit_image() -> None:
    view = snx.View(
        blending="default",
        scene=snx.Scene(
            children=[
                snx.Image(
                    data=np.random.randint(0, 255, (100, 100)).astype(np.uint8),
                ),
            ],
        ),
    )
    snx.show(view)

    tform = view.camera.transform
    assert tform == snx.Transform().translated((50, 50, 0))

    proj = view.camera.projection
    # FIXME: Is the z coordinate important?
    # FIXME: Remove atol
    # FIXME: Test entire matrix
    assert np.allclose(np.diag(proj)[:2], np.asarray([0.018, 0.018]), atol=1e-3)
