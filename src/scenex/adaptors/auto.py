from .registry import AdaptorRegistry


def get_adaptor_registry() -> AdaptorRegistry:
    """Get the backend adaptor registry."""
    from .pygfx import adaptors

    return adaptors
