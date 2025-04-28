from .registry import AdaptorRegistry


def get_adaptor_registry(backend: str | None = None) -> AdaptorRegistry:
    """Get the backend adaptor registry."""
    if backend not in (None, "pygfx"):
        raise ValueError(f"Unknown backend: {backend}")
    from .pygfx import adaptors

    return adaptors
