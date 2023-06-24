"""
"""

import warnings
from typing import Any, List, Dict

from ..nodes import ServerConfig, ClientConfig, Server, Client


_built_in_algorithms = {}


def register_algorithm(name: str) -> Any:
    """Decorator to register a new algorithm.

    Parameters
    ----------
    name : str
        Name of the algorithm.

    Returns
    -------
    The decorated class.

    """

    def wrapper(cls_: Any) -> Any:
        if issubclass(cls_, ServerConfig):
            field = "server_config"
        elif issubclass(cls_, ClientConfig):
            field = "client_config"
        elif issubclass(cls_, Server):
            field = "server"
        elif issubclass(cls_, Client):
            field = "client"
        else:
            raise ValueError(f"{cls_} is not a valid algorithm component")
        if name in _built_in_algorithms and field in _built_in_algorithms[name]:
            # raise ValueError(f"{name}.{field} has already been registered")
            warnings.warn(f"{name}.{field} has already been registered", RuntimeWarning)
        elif name not in _built_in_algorithms:
            _built_in_algorithms[name] = {}
        _built_in_algorithms[name][field] = cls_
        return cls_

    return wrapper


def list_algorithms() -> List[str]:
    return list(_built_in_algorithms)


def get_algorithm(name: str) -> Dict[str, Any]:
    if name not in _built_in_algorithms:
        raise ValueError(f"Algorithm {name} is not registered")
    return _built_in_algorithms[name]
