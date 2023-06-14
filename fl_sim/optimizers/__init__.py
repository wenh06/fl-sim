"""
"""

import importlib
import inspect
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Union, Any, Optional

import torch.optim as opt
import torch_optimizer as topt
from easydict import EasyDict as ED
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch_ecg.utils import add_docstring

from fl_sim.utils.misc import add_kwargs


# import optimizers in this folder dynamically
_local_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_local_dir.parent))
__all = []
for py_file in _local_dir.glob("*.py"):
    if py_file.name == "__init__.py":
        continue
    # import items in `__all__` in each file
    mod = importlib.import_module(f".{py_file.stem}", package=_local_dir.name)
    assert "__all__" in [
        name for name, obj in inspect.getmembers(mod)
    ], f"__all__ is not defined in {py_file}"
    __all.extend(mod.__all__)
    # add items in `__all__` to current namespace
    for item in mod.__all__:
        globals()[item] = getattr(mod, item)
del _local_dir, py_file, mod, item
sys.path.pop(0)


__all__ = [
    "get_optimizer",
    "get_inner_solver",
    "get_oracle",
    "available_optimizers",
    "available_optimizers_plus",
    "register_optimizer",
] + __all


del __all


_extra_kwargs = dict(
    local_weights=None,
    dual_weights=None,
    variance_buffer=None,
)


_available_optimizers = {
    obj.__name__: obj
    for obj in globals().values()
    if inspect.isclass(obj) and issubclass(obj, Optimizer) and obj != Optimizer
}

available_optimizers = list(_available_optimizers)
_extra_opt_optimizers = {
    obj_name: getattr(opt, obj_name)
    for obj_name in dir(opt)
    if eval(
        f"inspect.isclass(opt.{obj_name}) and issubclass(opt.{obj_name}, Optimizer) "
        f"and opt.{obj_name} != Optimizer"
    )
}
_extra_topt_optimizers = {
    obj_name: getattr(topt, obj_name)
    for obj_name in dir(topt)
    if eval(
        f"inspect.isclass(topt.{obj_name}) and issubclass(topt.{obj_name}, Optimizer) "
        f"and topt.{obj_name}.__name__ not in dir(opt) and topt.{obj_name} != Optimizer "
        f"and 'params' in inspect.getfullargspec(topt.{obj_name}).args"
    )
}
_available_optimizers_plus = {
    **_available_optimizers,
    **_extra_opt_optimizers,
    **_extra_topt_optimizers,
}
available_optimizers_plus = list(_available_optimizers_plus)


def register_optimizer(name: Optional[str] = None) -> type:
    """Decorator for registering custom optimizer.

    Parameters
    ----------
    name : Optional[str], optional
        name of the optimizer, by default None

    Returns
    -------
    type
        optimizer class

    """

    def wrapper(cls: type) -> type:
        if name is None:
            _name = cls.__name__
        else:
            _name = name
        assert issubclass(cls, Optimizer), f"{cls} is not a valid optimizer."
        assert (
            _name not in _available_optimizers_plus
        ), f"{_name} is already registered."
        _available_optimizers[_name] = cls
        available_optimizers.append(_name)
        _available_optimizers_plus[_name] = cls
        available_optimizers_plus.append(_name)
        return cls

    return wrapper


def get_optimizer(
    optimizer_name: Union[str, type],
    params: Iterable[Union[dict, Parameter]],
    config: Any,
) -> Optimizer:
    """Get optimizer by name.

    Parameters
    ----------
    optimizer_name : Union[str, type]
        optimizer name or class
    params : Iterable[Union[dict, Parameter]]
        parameters to be optimized
    config : Any
        config for optimizer

    Returns
    -------
    Optimizer
        Instance of the given optimizer.

    Examples
    --------
    .. code-block:: python

        import torch

        model = torch.nn.Linear(10, 1)
        optimizer = get_optimizer("SGD", model.parameters(), {"lr": 1e-2})  # PyTorch built-in
        optimizer = get_optimizer("yogi", model.parameters(), {"lr": 1e-2})  # from pytorch_optimizer
        optimizer = get_optimizer("FedPD_SGD", model.parameters(), {"lr": 1e-2})  # federated

    """
    if inspect.isclass(optimizer_name) and issubclass(optimizer_name, Optimizer):
        # the class is passed directly
        optimizer = optimizer_name(params, **_get_args(optimizer_name, config))
        step_args = inspect.getfullargspec(optimizer.step).args
        optimizer.step = add_kwargs(
            optimizer.step,
            **{k: v for k, v in _extra_kwargs.items() if k not in step_args},
        )
        # NOTE: if `optimizer` is passed into a scheduler, the scheduler will
        # wrap the `optimizer.step` method with `with_counter` which requires
        # the `step` method to be a bound method with `__self__` attribute.
        # So we need to add `_with_counter` to our wrapped `step` method to
        # prevent the scheduler from wrapping it again which will cause error.
        # Further, in the function `get_scheduler`, we will add
        # `scheduler.optimizer._step_count = 1` before returning the scheduler,
        # which suppresses the following warning:
        # ``Detected call of `lr_scheduler.step()` before `optimizer.step()`.``.
        # The risk is one has to check that scheduler.step() is called after
        # optimizer.step() in the training loop by himself.
        optimizer.step._with_counter = True
        return optimizer
    try:
        # try to use PyTorch built-in optimizer
        _config = _get_args(eval(f"opt.{optimizer_name}"), config)
        optimizer = eval(f"opt.{optimizer_name}(params, **_config)")
        # print(f"PyTorch built-in optimizer {optimizer_name} is used.")
        step_args = inspect.getfullargspec(optimizer.step).args
        optimizer.step = add_kwargs(
            optimizer.step,
            **{k: v for k, v in _extra_kwargs.items() if k not in step_args},
        )
        optimizer.step._with_counter = True
        # print(f"optimizer_name: {optimizer_name}")
        return optimizer
    except Exception:
        try:
            # try to use optimizer from torch_optimizer
            try:
                optimizer_cls = topt.get(optimizer_name)
            except ValueError:
                optimizer_cls = eval(f"topt.{optimizer_name}")
            optimizer = optimizer_cls(params, **_get_args(optimizer_cls, config))
            # print(f"Optimizer `{optimizer_name}` from torch_optimizer is used.")
            step_args = inspect.getfullargspec(optimizer.step).args
            optimizer.step = add_kwargs(
                optimizer.step,
                **{k: v for k, v in _extra_kwargs.items() if k not in step_args},
            )
            optimizer.step._with_counter = True
            return optimizer
        except Exception:
            pass

    if isinstance(config, dict):
        config = ED(config)

    built_in_optimizers = deepcopy(available_optimizers)

    # try to use federated local solver
    if optimizer_name not in available_optimizers:
        if f"{optimizer_name}Optimizer" in available_optimizers:
            # historical reason
            optimizer_name = f"{optimizer_name}Optimizer"
        else:
            # custom optimizer, added via `register_optimizer`
            optimizer_path = Path(optimizer_name).resolve()
            assert optimizer_path.exists() and optimizer_path.suffix == ".py", (
                f"Custom optimizer file `{optimizer_name}` does not exist, "
                "or it is not a python file."
            )
            spec = importlib.util.spec_from_file_location(
                optimizer_path.stem, optimizer_path
            )
            optimizer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(optimizer_module)
            # the custom algorithm should be added to the optimizer pool
            # using the decorator @register_optimizer
            new_optimizers = set(available_optimizers) - set(built_in_optimizers)
            assert len(new_optimizers) == 1, (
                f"Optimizer `{optimizer_name}` not found. "
                "Please check if the optimizer is registered using "
                "the decorator @register_optimizer from fl_sim.optimizers"
            )
            optimizer_name = new_optimizers.pop()
    assert optimizer_name in available_optimizers, (
        f"optimizer `{optimizer_name}` is not supported, "
        f"available optimizers are `{available_optimizers}`"
    )

    try:
        # optimizer = eval(
        #     f"{optimizer_name}(params, **_get_args({optimizer_name}, config))"
        # )
        optimizer_cls = _available_optimizers[optimizer_name]
        optimizer = optimizer_cls(params, **_get_args(optimizer_cls, config))
        # step_args = inspect.getfullargspec(optimizer.step).args
        # print(f"step_args: {step_args}")
        # if not set(_extra_kwargs).issubset(set(step_args)):
        #     optimizer.step = add_kwargs(
        #         optimizer.step,
        #         **{k: v for k, v in _extra_kwargs.items() if k not in step_args},
        #     )
        #     optimizer.step._with_counter = True
        return optimizer
    except Exception:
        raise ValueError(f"Optimizer `{optimizer_name}` is not supported.")


def _get_args(cls: type, config: Any) -> ED:
    """
    used to filter out the items in config that are not arguments of the class
    """
    if isinstance(config, dict):
        config = ED(config)
    args = [
        k
        for k in inspect.getfullargspec(cls.__init__).args
        if k
        not in [
            "self",
            "params",
        ]
    ]
    kwargs = ED()
    for k in args:
        try:
            kwargs[k] = eval(f"config.{k}")
        except Exception:
            pass
    return kwargs


# aliases


@add_docstring(
    get_optimizer.__doc__.replace("get optimizer", "get inner solver").replace(
        "optimizer = get_optimizer", "inner_solver = get_inner_solver"
    )
)
def get_inner_solver(
    optimizer_name: Union[str, type],
    params: Iterable[Union[dict, Parameter]],
    config: Any,
) -> Optimizer:
    return get_optimizer(optimizer_name, params, config)


@add_docstring(
    get_optimizer.__doc__.replace("get optimizer", "get oracle").replace(
        "optimizer = get_optimizer", "oracle = get_oracle"
    )
)
def get_oracle(
    optimizer_name: Union[str, type],
    params: Iterable[Union[dict, Parameter]],
    config: Any,
) -> Optimizer:
    return get_optimizer(optimizer_name, params, config)
