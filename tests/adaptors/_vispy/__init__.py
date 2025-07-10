"""Tests pertaining to VisPy adaptors"""

import pytest

from scenex.adaptors._auto import determine_backend

if not determine_backend() == "vispy":
    pytest.skip(
        "Skipping VisPy tests as VisPy will not be used in this environment",
        allow_module_level=True,
    )
