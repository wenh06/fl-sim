"""
"""

import os
import random
import re
import warnings
from collections import OrderedDict, defaultdict
from functools import wraps
from typing import Any, Callable, Union

import numpy as np
import torch

from .const import LOG_DIR


__all__ = [
    "experiment_indicator",
    "clear_logs",
    "ndarray_to_list",
    "ordered_dict_to_dict",
    "default_dict_to_dict",
    "set_seed",
]


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


# fmt: off
# numpy data types will be converted to python native types
_numpy_data_types = [
    "int_", "int", "intc", "intp", "int8", "int16", "int32", "int64", "uint8", "uint16",
    "uint32", "uint64", "float_", "float", "float16", "float32", "float64", "bool_", "bool",
    "short", "ushort", "uint", "uintc", "longlong", "ulonglong", "half", "single", "double",
    "longdouble", "csingle", "cdouble", "clongdouble", "object", "str_", "str", "unicode_",
    "unicode", "void", "byte", "ubyte", "complex_", "complex", "longcomplex", "datetime64",
    "timedelta64", "float128", "complex128", "complex256", "datetime64", "csingle", "cdouble",
    "clongdouble",
]
# fmt: on


with warnings.catch_warnings():
    # turn off possible DeprecationWarning from numpy
    warnings.filterwarnings("ignore")
    _numpy_data_types = tuple(
        getattr(np, item)
        for item in _numpy_data_types
        if hasattr(np, item) and "numpy" in str(getattr(np, item))
    )


def ndarray_to_list(x: Union[np.ndarray, dict, list, tuple]) -> Union[list, dict]:
    """Convert numpy array to list.

    This function is used to convert numpy array to list, so that it can be
    serialized by :mod:`json`.

    Parameters
    ----------
    x : Union[np.ndarray, dict, list, tuple]
        Input data, which can be numpy array,
        or dict, list, tuple containing numpy arrays.

    Returns
    -------
    Union[list, dict]
        Converted data.

    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, (list, tuple)):
        # to avoid cases where the list contains numpy data types
        return [ndarray_to_list(v) for v in x]
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = ndarray_to_list(v)
    elif isinstance(x, _numpy_data_types):
        return x.item()
    # the other types will be returned directly
    return x


def ordered_dict_to_dict(d: Union[OrderedDict, dict, list, tuple]) -> Union[dict, list]:
    """Convert ordered dict to dict.

    Parameters
    ----------
    d : Union[OrderedDict, dict, list, tuple]
        Input ordered dict,
        or dict, list, tuple containing ordered dicts.

    Returns
    -------
    Union[dict, list]
        Converted dict.

    """
    if isinstance(d, (OrderedDict, dict)):
        new_d = {}
        for k, v in d.items():
            new_d[k] = ordered_dict_to_dict(v)
    elif isinstance(d, (list, tuple)):
        new_d = [ordered_dict_to_dict(item) for item in d]
    else:
        new_d = d
    return new_d


def default_dict_to_dict(d: Union[defaultdict, dict, list, tuple]) -> Union[dict, list]:
    """Convert default dict to dict.

    Parameters
    ----------
    d : Union[defaultdict, dict, list, tuple]
        Input default dict,
        or dict, list, tuple containing default dicts.

    Returns
    -------
    Union[dict, list]
        Converted dict.

    """
    if isinstance(d, (defaultdict, dict)):
        new_d = {}
        for k, v in d.items():
            new_d[k] = default_dict_to_dict(v)
    elif isinstance(d, (list, tuple)):
        new_d = [default_dict_to_dict(item) for item in d]
    else:
        new_d = d
    return new_d


def set_seed(seed: int) -> None:
    """Set random seed for numpy and pytorch,
    as well as disable cudnn to ensure reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    None

    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
