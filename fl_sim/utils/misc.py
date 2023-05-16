"""
"""

import pathlib
from functools import wraps
from typing import Any, Callable


__all__ = [
    "PROJECT_DIR",
    "CACHED_DATA_DIR",
    "LOG_DIR",
    "experiment_indicator",
]


PROJECT_DIR = pathlib.Path(__file__).absolute().parents[2]

CACHED_DATA_DIR = PROJECT_DIR / ".data_cache"

LOG_DIR = PROJECT_DIR / ".logs"


CACHED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def experiment_indicator(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print("\n" + "-" * 100)
            print(f"  Start experiment {name}  ".center(100, "-"))
            print("-" * 100 + "\n")
            func(*args, **kwargs)
            print("\n" + "-" * 100)
            print(f"  End experiment {name}  ".center(100, "-"))
            print("-" * 100 + "\n")

        return wrapper

    return decorator
