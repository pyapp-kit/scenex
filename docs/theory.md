---
icon: material/lightbulb-outline
---
# Theory

`scenex` is a library for building **declarative** visuals, where users describe **what** is in the scene rather than **how** to render it. This allows the same scene code to run unchanged across environments - standalone or embedded in an existing application - insulated from changes in graphics technology.

---

## `scenex.models`

The module `scenex.models` contains [scenegraph](https://en.wikipedia.org/wiki/Scene_graph) building blocks, written as [pydantic](https://pydantic.dev/) dataclasses augmented with [psygnal](https://psygnal.readthedocs.io/) signals. This combination results in dataclasses with some useful properties:

- **Validation**: node properties are checked on assignment - for example:
    - `opacity` can be confined to `[0, 1]`, so bad values raise an error immediately rather than producing subtle visual artifacts.
    - sizes and widths can be confined to non-negative values.
- **Serialization**: scenegraphs can be serialized to/from JSON, making it straightforward to save or share a complete scenegraph.
- **Observability**: every field change emits a signal that any code can subscribe to, useful for keeping a UI control in sync with the scene or building a node that responds to changes in another.

These dataclasses are composed to describe **what** is in the scene using trees of `Node`s nested in a `Scene`, through the `View` that frames them, up to the `Canvas` they are rendered on.

``` mermaid
graph TD
    A[Canvas] --> B[View]
    B --> C[Scene]
    B --> D[Camera]
    C --> D
    C --> E[Image]
    C --> F[Points]
```

!!! note "on `View.layout`"
    Our use cases suggest users either want to position and size views based on **pixels** and/or **fractions**. As such, the `View.layout` model describes view bounds using `Coord`s, which can be fractional, absolute, or a mix of both:

    ```python
    view.layout.x = "0%", "50%"      # Take up the left half of the screen
    view.layout.y = "-40px", "100%"  # Take up the bottom 40 pixels of the screen
    ```
    
    By default, views cover the entire canvas - the common case of a single full-canvas view requires no layout configuration at all.

For a detailed list of the different `Node` types in `scenex`, check out the [API reference](references.md).

## `scenex.adaptors`

`scenex.adaptors` renders `scenex` models into pixels. When a model is first displayed, the chosen adaptor initializes its graphics toolkit and subscribes to the model's observable fields; subsequent changes to the model flow through automatically. By selecting a different adaptor, `scenex` can leverage different visualization technologies without changing the model. Two are currently supported:

| Adaptor | Graphics API | When to use |
|---------|-------------|-------------|
| `pygfx` | WebGPU (wgpu) | Default; modern hardware, best feature coverage |
| `vispy` | OpenGL      | Older hardware or environments without WebGPU |

In practice, users should use `scenex.show(obj)` instead of `scenex.adaptors`, as it also handles adaptor selection, adaptor initialization, and camera setup automatically.

!!! note
    To render a scene to an array instead of the screen, use `Canvas.render()`.

## `scenex.app`

Many visualizations are interactive - responding to mouse clicks, keyboard input, or live data updates. Interactivity depends on an event loop: `scenex` can start one itself, or integrate with one already provided by the host application. Every GUI framework - [Qt](https://riverbankcomputing.com/software/pyqt), [wxPython](https://wxpython.org), [Jupyter](https://jupyter.org) - has its own event loop, and they cannot coexist.

`scenex.app` contains the code to start new event loops, detect existing ones, and integrate into them to deliver user events and update pixels. It presents a uniform API for these events, allowing event code to be reused across any GUI framework. For users who don't need to embed their canvas in an existing application, `scenex.run()` starts a blocking event loop.
