"""
"""

import re
import warnings
from typing import Any, List, Dict, Optional

from ..nodes import ServerConfig, ClientConfig, Server, Client


_built_in_algorithms = {}


def register_algorithm(name: Optional[str] = None, override: bool = True) -> Any:
    """Decorator to register a new algorithm.

    Parameters
    ----------
    name : str, optional
        Name of the algorithm.
        If not specified, the class name with "(Server|Client)(?:Config)?"
        removed will be used.
    override : bool, default True
        Whether to override the existing algorithm with the same name.

    Returns
    -------
    The decorated class.

    """

    def wrapper(cls_: Any) -> Any:
        if name is None:
            if hasattr(cls_, "__name__"):
                _name = cls_.__name__
            else:
                _name = cls_.__class__.__name__
            _name = re.sub("(Server|Client)(?:Config)?$", "", _name)
        else:
            _name = name
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
        if _name in _built_in_algorithms and field in _built_in_algorithms[_name]:
            # raise ValueError(f"{_name}.{field} has already been registered")
            if not override:
                warnings.warn(
                    f"{_name}.{field} has already been registered", RuntimeWarning
                )
        elif _name not in _built_in_algorithms:
            _built_in_algorithms[_name] = {}
        if override or field not in _built_in_algorithms[_name]:
            _built_in_algorithms[_name][field] = cls_
        return cls_

    return wrapper


def list_algorithms() -> List[str]:
    return list(_built_in_algorithms)


def get_algorithm(name: str) -> Dict[str, Any]:
    if name not in _built_in_algorithms:
        raise ValueError(f"Algorithm {name} is not registered")
    return _built_in_algorithms[name]
