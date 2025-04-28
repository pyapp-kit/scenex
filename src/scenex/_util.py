import atexit
import os
import shutil
import tempfile
from functools import cache

import numpy as np


@cache
def _session_dir() -> str:
    """Get the temporary directory for the session."""
    tmpdir = tempfile.gettempdir()
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
    return tmpdir


def tmp_image(img: np.ndarray, tmp_name: str = "temp_image.png") -> str:
    """Create a temporary image file for the session."""
    import imageio.v3 as iio

    filename = os.path.join(_session_dir(), tmp_name)
    iio.imwrite(filename, img)
    return filename


def view_in_browser(
    img: np.ndarray, tmp_name: str = "temp_image.png", block: bool = False
) -> None:
    """Display an image in the default web browser."""
    import webbrowser

    webbrowser.open("file://" + tmp_image(img, tmp_name))

    if block:
        input("Press Enter to continue. (This will delete the image.)")
