"""Auto-detection of the backend adaptor registry."""

from ._registry import AdaptorRegistry


def get_adaptor_registry(backend: str | None = None) -> AdaptorRegistry:
    """Get the backend adaptor registry."""
    if backend not in ("pygfx", None):
        raise ValueError(f"Unknown backend: {backend}")

    from ._pygfx import adaptors

    return adaptors
