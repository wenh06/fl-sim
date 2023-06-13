"""
"""

from typing import Any, List, Dict

from ..nodes import ServerConfig, ClientConfig, Server, Client


_built_in_algorithms = {}


def register_algorithm(name: str) -> Any:
    def wrapper(cls: Any) -> Any:
        if issubclass(cls, ServerConfig):
            field = "server_config"
        elif issubclass(cls, ClientConfig):
            field = "client_config"
        elif issubclass(cls, Server):
            field = "server"
        elif issubclass(cls, Client):
            field = "client"
        else:
            raise ValueError(f"{cls} is not a valid algorithm component")
        if name in _built_in_algorithms and field in _built_in_algorithms[name]:
            raise ValueError(f"{name}.{field} has already been registered")
        elif name not in _built_in_algorithms:
            _built_in_algorithms[name] = {}
        _built_in_algorithms[name][field] = cls
        return cls

    return wrapper


def list_algorithms() -> List[str]:
    return list(_built_in_algorithms)


def get_algorithm(name: str) -> Dict[str, Any]:
    if name not in _built_in_algorithms:
        raise ValueError(f"Algorithm {name} is not registered")
    return _built_in_algorithms[name]
