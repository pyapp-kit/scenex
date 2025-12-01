"""Camera controllers have been moved to scenex.model.

This module previously contained imperative controller implementations.
Controllers are now declarative pydantic models in :mod:`scenex.model`:

- Use :class:`scenex.PanZoomController` for 2D pan/zoom interactions
- Use :class:`scenex.OrbitController` for 3D orbit interactions

Example
-------
>>> import scenex as snx
>>> camera = snx.Camera(controller=snx.PanZoomController(), interactive=True)
"""
