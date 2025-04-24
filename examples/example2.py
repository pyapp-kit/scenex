import numpy as np

import scenex as snx
from scenex.adaptors.vispy._adaptor_registry import get_adaptor
from qtpy.QtWidgets import QApplication
from vispy import scene

qapp = QApplication([])
canvas = scene.SceneCanvas(keys="interactive", size=(800, 600))
# view = canvas.central_widget.add_view()

# here we make a scenex image and scene
X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
sine_img = (np.sin(X) * np.cos(Y)).astype(np.float32)
image = snx.Image(name="sine image", data=sine_img, clims=(-1, 1))
snx_scene = snx.Scene(children=[image])
our_camera = snx.Camera()
our_view = snx.View(scene=snx_scene, camera=our_camera)

# # convert to a vispy object and add to our view
vis_image_node = get_adaptor(image)._snx_get_native()
# vis_scene = get_adaptor(snx_scene)._snx_get_native()
# vis_our_camera = get_adaptor(our_camera)._snx_get_native()
# vis_our_view = get_adaptor(our_view)._snx_get_native()

# canvas.central_widget.add_widget(vis_our_view)

# canvas.show()
# qapp.exec()
