---
icon: material/lightbulb-outline
---
# Theory

`scenex` is a library for building visuals **declaratively**. Users describe **what** is in the scene, but *not* **how** to render it - allowing the same scene to run in any environment - standalone or embedded in an existing application - and insulating it from changes in graphics technology.

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

### `scenex.app.events`

A scene isn't fully described by its visual state alone - how it evolves in response to user input is part of what the scene *is*, and expressing that declaratively would be a natural completion of `scenex`'s design. **No general solution to this has been found yet**: updating a text label when the user mouses over a point, for instance, is far easier to express as an algorithm than as a declarative specification. There is also a structural problem, as some fields only make sense in context of others; an orbit center has no meaning unless the camera controller is set to orbit, and encoding that dependency cleanly in a declarative model is awkward.

For these reasons, interactive behaviors in `scenex` are expressed imperatively. Where common behaviors repeat across scenes, they are wrapped in pydantic model classes - camera controllers, view resizers - that act on the declarative scene nodes. Users declare *which* behavior they want as a model field, keeping the scene description serializable even where the underlying logic is imperative.

For custom interactions that don't fit a reusable pattern (like the text-point interaction described above), `scenex` provides event filters: callbacks registered on a model that receive events and return whether the event was handled. Filters are an intentional escape hatch - they drop into imperative code where declarative description isn't practical.

Where a filter should live follows from the nature of the event. Mouse events carry a canvas position and are naturally handled at the `View` level. Because `scenex` scenes are always 3D - even when displaying 2D data - custom mouse interaction code typically centers on rays: the view has the camera context needed to unproject a 2D canvas position into a world-space ray, and that ray is the natural representation for querying which objects are under the cursor. Keyboard events carry no position and can't be routed to a particular view, so they belong at the `Canvas` level.

!!! note "Why no node-level filters?"
    Event filters are intentionally not placed on individual nodes. Part of this is to minimize API in the absence of a compelling use case; part is that node-level routing would require careful design to preserve performance - computing intersections against complex geometry on every event is not something you want to do naively.
