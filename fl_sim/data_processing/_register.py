import warnings
from typing import Any, List, Optional

from .fed_dataset import FedDataset


_built_in_fed_datasets = {}


def register_fed_dataset(name: Optional[str] = None, override: bool = True) -> Any:
    """Decorator to register a new federated dataset.

    Parameters
    ----------
    name : str, optional
        Name of the federated dataset.
        If not specified, the class name will be used.
    override : bool, default True
        Whether to override the existing federated dataset with the same name.

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
        else:
            _name = name
        assert issubclass(cls_, FedDataset), f"{cls_} is not a valid dataset"
        if _name in _built_in_fed_datasets:
            if override:
                _built_in_fed_datasets[_name] = cls_
            else:
                # raise ValueError(f"{_name} has already been registered")
                warnings.warn(f"{_name} has already been registered", RuntimeWarning)
        else:
            _built_in_fed_datasets[_name] = cls_
        return cls_

    return wrapper


def list_fed_dataset() -> List[str]:
    """List all registered federated datasets."""
    return list(_built_in_fed_datasets)


def get_fed_dataset(name: str) -> Any:
    """Get a registered federated dataset by name."""
    if name not in _built_in_fed_datasets:
        raise ValueError(f"Federated dataset {name} is not registered")
    return _built_in_fed_datasets[name]
