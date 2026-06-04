---
icon: material/home
---
# scenex

`scenex` is a library for describing and visualizing scenes, with minimal dependencies.

<div class="grid cards cols-3" markdown>

-   :material-pencil-ruler:{ .lg .middle } **Declarative**

    ---

    Scenes are created by [pydantic](https://pydantic.dev/) models that say *what* to show instead of *how* to show it.

-   :material-flash:{ .lg .middle } **Evented**

    ---

    Nodes can react to changes throughout the scene — update your data, and the scene updates with it.

-   :material-puzzle:{ .lg .middle } **Flexible**

    ---

    Visualize with [vispy](https://vispy.org/) or [pygfx](https://docs.pygfx.org/stable/index.html), rendered to widgets in [qtpy](https://github.com/spyder-ide/qtpy), [jupyter](https://jupyter.org/), or [wx](https://wxpython.org/index.html)

</div>


---

!!! warning "In development"

    scenex is a work in progress.  The public API may change between releases.

## Installation

Because `scenex` can run with different visualization and widget backends, it doesn't ship with any backends by default. You can install the backends you'll like to use with extras:

=== "pygfx + Qt"

    ```bash
    pip install "scenex[pygfx,pyqt6]"
    ```

=== "vispy + Jupyter"

    ```bash
    pip install "scenex[pygfx,jupyter]"
    ```

See the [install istructions](install.md) for comprehensive installation instructions!

## Usage

```python
import numpy as np
import scenex as snx

# A single node — show() wraps it in a Scene and View for you
data = np.random.rand(100, 100).astype(np.float32)
img  = snx.Image(data=data)

snx.show(img)
snx.run()            # enter the event loop (not needed in Jupyter)
```

Mutating the model *after* it is displayed updates the rendered scene
immediately:

```python
from cmap import Colormap

img.cmap   = Colormap("viridis")
img.clims  = (0.2, 0.8)
img.opacity = 0.7
```
