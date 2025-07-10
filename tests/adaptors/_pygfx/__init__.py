"""Tests pertaining to Pygfx components"""

import pytest

from scenex.adaptors._auto import determine_backend

if not determine_backend() == "pygfx":
    pytest.skip(
        "Skipping Pygfx tests as Pygfx will not be used in this environment",
        allow_module_level=True,
    )
