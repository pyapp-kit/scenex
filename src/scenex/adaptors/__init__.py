"""Backend adaptors that translate scenex models into graphics library calls.

Adaptors bridge the gap between scenex's declarative models and rendering backends
(e.g. pygfx). For each model class in `scenex.model`, there should be a corresponding
adaptor class for each backend that handles the actual GPU rendering.

When you call `scenex.show()`, adaptors are automatically created for each object
in your scene graph. The adaptors translate model properties (colors, transforms,
data) into backend-specific commands. As you modify the models, events trigger updates
in the adaptors to keep the rendered scene synced.

Architecture
------------
The adaptor system uses a registry pattern::

    Model (declarative) → Adaptor (imperative) → Backend (GPU library)

    Image model         → ImageAdaptor         → pygfx.Mesh + texture
                        or
                        → ImageAdaptor         → vispy.scene.Image

Each backend (pygfx, vispy) has its own set of adaptors implementing the same
model-to-native translation logic tailored to that backend's API.

Main Components
---------------
- AdaptorRegistry: Maps model classes to adaptor classes
- Adaptor: Base class for all adaptors, handles event subscription
- Backend-specific adaptors: In _pygfx and _vispy subpackages

Supported Backends
------------------
**pygfx** (WebGPU-based):
    - Modern GPU API with advanced rendering features
**vispy** (OpenGL-based):
    - Mature, widely-supported OpenGL renderer

Usage
-----
Adaptors are not intended for manual instantiation; they are instead created
automatically by `scenex.show()`::

    import scenex as snx

    # Create model
    scene = snx.Scene(children=[snx.Image(data=my_array)])

    # This creates adaptors automatically
    snx.show(scene)  # Adaptors sync model to GPU
    snx.run()

To select a particular backend, use `scenex.use()`::

    snx.use("pygfx")  # or "vispy"
    snx.show(scene)

See Also
--------
scenex.model : Declarative model classes
scenex.use : Function to select rendering backend
scenex.show : Function that creates adaptors
"""

from ._auto import get_adaptor_registry, get_all_adaptors, run, use
from ._base import Adaptor
from ._registry import AdaptorRegistry

__all__ = [
    "Adaptor",
    "AdaptorRegistry",
    "get_adaptor_registry",
    "get_all_adaptors",
    "run",
    "use",
]
