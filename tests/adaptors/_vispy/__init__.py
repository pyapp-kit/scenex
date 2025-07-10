"""Tests pertaining to VisPy adaptors"""

from importlib.util import find_spec

import pytest

if not find_spec("vispy"):
    pytest.skip(
        "Skipping vispy tests as vispy is not installed", allow_module_level=True
    )
