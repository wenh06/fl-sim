"""
"""

import pathlib
import re
from functools import wraps
from typing import Any, Callable


__all__ = [
    # "PROJECT_DIR",
    "CACHED_DATA_DIR",
    "LOG_DIR",
    "experiment_indicator",
    "clear_logs",
]


PROJECT_DIR = pathlib.Path(__file__).absolute().parents[2]

USER_CACHE_DIR = pathlib.Path.home() / ".cache" / "fl-sim"

CACHED_DATA_DIR = USER_CACHE_DIR / ".data_cache"

LOG_DIR = USER_CACHE_DIR / ".logs"


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


def clear_logs(pattern: str = "*") -> None:
    """Clear given log files in LOG_DIR.

    Parameters
    ----------
    pattern : str, optional
        Pattern of log files to be cleared, by default "*"
        The searching will be executed by :func:`re.search`.

    """
    for log_file in [fp for fp in LOG_DIR.glob("*") if re.search(pattern, fp.name)]:
        log_file.unlink()
