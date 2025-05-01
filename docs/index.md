# scenex

!!! warning "In development"

    This library is a work in progress. The API will change frequently
    as we add new features and improve existing ones.

Scenex is a Python API for creating and manipulating 3D scenes.

It does not implement any rendering or graphics directly, but rather serves
as a high-level interface and adaptor for existing scene-graph libraries,
such as [vispy](https://vispy.org/) and [pygfx](https://pygfx.org/), and
hopefully others (like [datovis](https://datoviz.org/)) in the future.

The goal is to provide a clear scene graph model (backed by [pydantic](https://docs.pydantic.dev)
), with backend adaptors that connect the model to the actual rendering
engine.  The models emit events upon mutation (using [psygnal](https://psygnal.readthedocs.io)),
and the adaptors listen to these events and update the scene graph.

Because the models are backed by pydantic, they can be easily serialized to JSON
and other formats, making it easy to save and load scenes, and define them
declaratively.
