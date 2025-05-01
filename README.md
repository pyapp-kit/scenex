# scenex

[![License](https://img.shields.io/pypi/l/scenex.svg?color=green)](https://github.com/pyapp-kit/scenex/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/scenex.svg?color=green)](https://pypi.org/project/scenex)
[![Python Version](https://img.shields.io/pypi/pyversions/scenex.svg?color=green)](https://python.org)
[![CI](https://github.com/pyapp-kit/scenex/actions/workflows/ci.yml/badge.svg)](https://github.com/pyapp-kit/scenex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pyapp-kit/scenex/branch/main/graph/badge.svg)](https://codecov.io/gh/pyapp-kit/scenex)

*Declarative, reactive scene graph model, with backend adapter abstraction*

---------

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
