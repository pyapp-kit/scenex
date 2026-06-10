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

## Alternatives

Like any tool, `scenex` is not a panacea. It aims to be **flexible**, describing scenes at a high level and abstracting away graphics primitives, animation loops, etc. *If you need lower-level access, consider using [pygfx](https://pygfx.org), [vispy](https://vispy.org), or [datoviz](https://datoviz.org/) directly.*

The primary focus of `scenex` is rendering **scenes**. While you can render *anything* in a scene, you might get more convenience from alternative visualization tools:

| I want to visualize... | Consider |
|-|-|
| plots | [matplotlib](https://matplotlib.org), [fastplotlib](https://www.fastplotlib.org/), [Plotly](https://plotly.com/python/), [Bokeh](https://bokeh.org), [Vega-Altair](https://altair-viz.github.io/) |
| nD datasets | [napari](https://napari.org) |
| 3D meshes / geometry | [PyVista](https://pyvista.org), [vedo](https://vedo.embl.es) |
