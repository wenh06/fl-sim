"""
"""

import inspect
import os
import random
import re
import types
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Union, Optional, Sequence

import numpy as np
import torch
from torch_ecg.utils import get_kwargs, get_required_args

from .const import LOG_DIR


__all__ = [
    "experiment_indicator",
    "clear_logs",
    "ndarray_to_list",
    "ordered_dict_to_dict",
    "default_dict_to_dict",
    "set_seed",
    "get_scheduler",
    "get_scheduler_info",
    "is_notebook",
    "find_longest_common_substring",
    "add_kwargs",
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


def get_scheduler(
    scheduler_name: str, optimizer: torch.optim.Optimizer, config: Optional[dict]
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler.

    Parameters
    ----------
    scheduler_name : str
        Name of the scheduler.
    optimizer : torch.optim.Optimizer
        Optimizer.
    config : dict
        Configuration of the scheduler.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.

    """
    scheduler_name = scheduler_name.lower()
    config = {} if config is None else config

    if scheduler_name == "none":
        if config:
            warnings.warn(
                "Scheduler is not used, but config is provided. "
                "The config will be ignored.",
                RuntimeWarning,
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1.0
        )
        scheduler.optimizer._step_count = 1  # to prevent scheduler warning
        return scheduler

    if scheduler_name == "cosine":
        scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR
    elif scheduler_name == "step":
        scheduler_cls = torch.optim.lr_scheduler.StepLR
    elif scheduler_name == "multi_step":
        scheduler_cls = torch.optim.lr_scheduler.MultiStepLR
    elif scheduler_name == "exponential":
        scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
    elif scheduler_name == "reduce_on_plateau":
        scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    elif scheduler_name == "cyclic":
        scheduler_cls = torch.optim.lr_scheduler.CyclicLR
    elif scheduler_name == "one_cycle":
        scheduler_cls = torch.optim.lr_scheduler.OneCycleLR
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    defaul_configs = get_kwargs(scheduler_cls)
    required_configs = get_required_args(scheduler_cls)
    required_configs.remove("optimizer")
    if "self" in required_configs:
        # normally, "self" is not in required_configs
        # just in case
        required_configs.remove("self")
    assert set(required_configs).issubset(set(config.keys())), (
        f"Missing required configs for {scheduler_name}: "
        f"{set(required_configs) - set(config.keys())}"
    )
    assert (set(config.keys()) - set(required_configs)).issubset(
        set(defaul_configs.keys())
    ), (
        f"Unsupported configs for {scheduler_name}: "
        f"{set(config.keys()) - set(required_configs) - set(defaul_configs.keys())}"
    )
    scheduler_config = defaul_configs.copy()
    scheduler_config.update(config)
    scheduler = scheduler_cls(optimizer, **scheduler_config)
    scheduler.optimizer._step_count = 1  # to prevent scheduler warning

    return scheduler


def get_scheduler_info(scheduler_name: str) -> dict:
    """Get information of the scheduler,
    including the required and optional configs.
    """
    if scheduler_name == "cosine":
        scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR
    elif scheduler_name == "step":
        scheduler_cls = torch.optim.lr_scheduler.StepLR
    elif scheduler_name == "multi_step":
        scheduler_cls = torch.optim.lr_scheduler.MultiStepLR
    elif scheduler_name == "exponential":
        scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
    elif scheduler_name == "reduce_on_plateau":
        scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    elif scheduler_name == "cyclic":
        scheduler_cls = torch.optim.lr_scheduler.CyclicLR
    elif scheduler_name == "one_cycle":
        scheduler_cls = torch.optim.lr_scheduler.OneCycleLR
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    defaul_configs = get_kwargs(scheduler_cls)
    required_configs = get_required_args(scheduler_cls)
    required_configs.remove("optimizer")
    if "self" in required_configs:
        # normally, "self" is not in required_configs
        # just in case
        required_configs.remove("self")

    return {
        "class": scheduler_cls,
        "required_args": required_configs,
        "optional_args": defaul_configs,
    }


def is_notebook() -> bool:
    """Check if the current environment is a notebook (Jupyter or Colab).

    Implementation adapted from [#sa]_.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        Whether the code is running in a notebook

    References
    ----------
    .. [#sa] https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook

    """
    try:
        shell = get_ipython().__class__
        if shell.__name__ == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif "colab" in repr(shell).lower():
            return True  # Google Colab
        elif shell.__name__ == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:  # Other type (?)
            return False
    except NameError:  # Probably standard Python interpreter
        return False
    except TypeError:  # get_ipython is None
        return False


def find_longest_common_substring(
    strings: Sequence[str], min_len: Optional[int] = None, ignore: Optional[str] = None
) -> str:
    """Find the longest common substring of a list of strings.

    Parameters
    ----------
    strings : sequence of str
        The list of strings.
    min_len : int, optional
        The minimum length of the common substring.
    ignore : str, optional
        The substring to be ignored.

    Returns
    -------
    str
        The longest common substring.

    """
    substr = ""
    if len(strings) > 1 and len(strings[0]) > 0:
        for i in range(len(strings[0])):
            for j in range(len(strings[0]) - i + 1):
                if j > len(substr) and all(strings[0][i : i + j] in x for x in strings):
                    substr = strings[0][i : i + j]
    if ignore is not None and ignore in substr:
        substr = max(substr.split(ignore), key=len)
    if min_len is not None and len(substr) < min_len:
        return ""
    return substr


def add_kwargs(func: callable, **kwargs: Any) -> callable:
    """Add keyword arguments to a function.

    This function is used to add keyword arguments to a function
    in order to make it compatible with other functions。

    Parameters
    ----------
    func : callable
        The function to be decorated.
    kwargs : dict
        The keyword arguments to be added.

    Returns
    -------
    callable
        The decorated function, with the keyword arguments added.

    """
    old_kwargs = get_kwargs(func)
    func_signature = inspect.signature(func)
    func_parameters = func_signature.parameters.copy()  # ordered dict

    full_kwargs = deepcopy(old_kwargs)
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    for k, v in func_parameters.items():
        if v.kind == inspect.Parameter.KEYWORD_ONLY:
            kind = inspect.Parameter.KEYWORD_ONLY
            break

    for k, v in kwargs.items():
        if k in old_kwargs:
            raise ValueError(f"keyword argument `{k}` already exists!")
        full_kwargs[k] = v
        func_parameters[k] = inspect.Parameter(k, kind, default=v)

    # move the VAR_POSITIONAL and VAR_KEYWORD in `func_parameters` to the end
    for k, v in func_parameters.items():
        if v.kind == inspect.Parameter.VAR_POSITIONAL:
            func_parameters.move_to_end(k)
            break
    for k, v in func_parameters.items():
        if v.kind == inspect.Parameter.VAR_KEYWORD:
            func_parameters.move_to_end(k)
            break

    if isinstance(func, types.MethodType):
        # can not assign `__signature__` to a bound method directly
        func.__func__.__signature__ = func_signature.replace(
            parameters=func_parameters.values()
        )
    else:
        func.__signature__ = func_signature.replace(parameters=func_parameters.values())

    # docstring is automatically copied by `functools.wraps`

    @wraps(func)
    def wrapper(*args: Any, **kwargs_: Any) -> Any:
        assert set(kwargs_).issubset(full_kwargs), (
            "got unexpected keyword arguments: "
            f"{list(set(kwargs_).difference(full_kwargs))}"
        )
        filtered_kwargs = {k: v for k, v in kwargs_.items() if k in old_kwargs}
        return func(*args, **filtered_kwargs)

    return wrapper
