"""
"""

from typing import Any, List, Optional

from .fed_dataset import FedDataset


_built_in_fed_datasets = {}


def register_fed_dataset(name: Optional[str] = None) -> Any:
    """Decorator to register a new federated dataset.

    Parameters
    ----------
    name : str, optional
        Name of the federated dataset.
        If not specified, the class name will be used.

    Returns
    -------
    The decorated class.

    """

    def wrapper(cls: Any) -> Any:
        if name is None:
            _name = cls.__name__
        else:
            _name = name
        assert issubclass(cls, FedDataset), f"{cls} is not a valid dataset"
        if _name in _built_in_fed_datasets:
            raise ValueError(f"{_name} has already been registered")
        _built_in_fed_datasets[_name] = cls
        return cls

    return wrapper


def list_fed_dataset() -> List[str]:
    return list(_built_in_fed_datasets)


def get_fed_dataset(name: str) -> Any:
    if name not in _built_in_fed_datasets:
        raise ValueError(f"Federated dataset {name} is not registered")
    return _built_in_fed_datasets[name]
