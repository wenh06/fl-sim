"""
"""

import re
import warnings
from typing import Any, List, Dict, Optional

import torch.optim as optim


_built_in_optimizers = {}


def register_optimizer(name: Optional[str] = None) -> Any:
    """Decorator to register a new optimizer.

    Parameters
    ----------
    name : str, optional
        Name of the optimizer.
        If not specified, the class name with "(?:Optimizer)?"
        removed will be used.

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
            _name = re.sub("(?:Optimizer)?$", "", _name)
        else:
            _name = name
        assert issubclass(cls_, optim.Optimizer), f"{cls_} is not a valid optimizer"
        if _name in _built_in_optimizers:
            # raise ValueError(f"{_name} has already been registered")
            warnings.warn(f"{_name} has already been registered", RuntimeWarning)
        _built_in_optimizers[_name] = cls_
        return cls_

    return wrapper


def list_optimizers() -> List[str]:
    return list(_built_in_optimizers)


def get_optimizer(name: str) -> Dict[str, Any]:
    if name not in _built_in_optimizers:
        _name = re.sub("(?:Optimizer)?$", "", name)
    else:
        _name = name
    if _name not in _built_in_optimizers:
        raise ValueError(f"Optimizer {name} is not registered")
    return _built_in_optimizers[_name]
