"""
Utilities for dynamic imports.
"""

import importlib
import types
from pathlib import Path
from typing import Union


_all__ = [
    "load_module_from_file",
]


def load_module_from_file(file_path: Union[str, Path]) -> types.ModuleType:
    """Load a module from a file.

    Parameters
    ----------
    file_path : str or Path
        The path of the file.

    Returns
    -------
    types.ModuleType
        The loaded module.

    """
    file_path = Path(file_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
