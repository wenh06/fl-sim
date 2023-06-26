"""
"""

import inspect
from pathlib import Path
from typing import Iterable, Union, Any

import torch.optim as opt
import torch_optimizer as topt
from easydict import EasyDict as ED
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch_ecg.utils import add_docstring

from ..utils.misc import add_kwargs
from ..utils.imports import load_module_from_file
from ._register import (  # noqa: F401
    list_optimizers as list_builtin_optimizers,
    get_optimizer as get_builtin_optimizer,
    register_optimizer,
)  # noqa: F401
from . import (  # noqa: F401
    base,
    feddr,
    fedpd,
    fedprox,
    pfedmac,
    pfedme,
    scaffold,
)  # noqa: F401


__all__ = [
    "get_optimizer",
    "get_inner_solver",
    "get_oracle",
    "available_optimizers",
    "available_optimizers_plus",
    "register_optimizer",
]


_extra_kwargs = dict(
    local_weights=None,
    dual_weights=None,
    variance_buffer=None,
)


_available_optimizers = {
    item: get_builtin_optimizer(item) for item in list_builtin_optimizers()
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


def get_optimizer(
    optimizer_name: Union[str, type],
    params: Iterable[Union[dict, Parameter]],
    config: Any,
) -> Optimizer:
    """Get optimizer by name.

    Parameters
    ----------
    optimizer_name : Union[str, type]
        Optimizer name or class
    params : Iterable[Union[dict, Parameter]]
        Parameters to be optimized
    config : Any
        Config for optimizer.
        Should be a dict or a class with attributes
        which can be accessed by `config.attr`.

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
        optimizer = optimizer_name(params, **_get_cls_init_args(optimizer_name, config))
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
        _config = _get_cls_init_args(eval(f"opt.{optimizer_name}"), config)
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
            optimizer = optimizer_cls(
                params, **_get_cls_init_args(optimizer_cls, config)
            )
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
        # convert dict to easydict so that we can use dot notation
        # to access config items in function `_get_cls_init_args`
        # like items in `ClientConfig` can be accessed by `config.xxx`
        config = ED(config)

    builtin_optimizers = list_builtin_optimizers().copy()

    # try to use federated local solver
    if optimizer_name not in builtin_optimizers:
        if f"{optimizer_name}Optimizer" in builtin_optimizers:
            # historical reason
            optimizer_name = f"{optimizer_name}Optimizer"
        else:
            # custom optimizer, added via `register_optimizer`
            optimizer_file = Path(optimizer_name).expanduser().resolve()
            if optimizer_file.suffix == ".py":
                # is a .py file
                # in this case, there should be only one optimizer class registered in the file
                optimizer_name = None
            else:
                # of the form /path/to/opt_file_stem.opt_name
                # in this case, there could be multiple optimizers registered in the file
                optimizer_file, optimizer_name = str(optimizer_file).rsplit(".", 1)
                optimizer_file = Path(optimizer_file + ".py").expanduser().resolve()
            assert optimizer_file.exists(), (
                f"Optimizer `{optimizer_file}` not found. "
                "Please check if the optimizer file exists and is a .py file, "
                "or of the form ``/path/to/opt_file_stem.opt_name``"
            )
            optimizer_module = load_module_from_file(optimizer_file)
            # the custom algorithm should be added to the optimizer pool
            # using the decorator @register_optimizer
            new_optimizers = [
                item
                for item in get_builtin_optimizer()
                if item not in builtin_optimizers
            ]
            if optimizer_name is None:
                if len(new_optimizers) == 0:
                    raise ValueError(
                        f"No optimizer found in `{optimizer_file}`. "
                        "Please check if the optimizer is registered using "
                        "the decorator ``@register_optimizer`` from ``fl_sim.optimizers``"
                    )
                elif len(new_optimizers) > 1:
                    raise ValueError(
                        f"Multiple optimizers found in `{optimizer_file}`. "
                        "Please split the optimizers into different files, "
                        "or pass the optimizer name in the form "
                        "``/path/to/opt_file_stem.opt_name``"
                    )
                optimizer_name = new_optimizers[0]
            else:
                if optimizer_name not in new_optimizers:
                    raise ValueError(
                        f"Optimizer `{optimizer_name}` not found in `{optimizer_file}`. "
                        "Please check if the optimizer is registered using "
                        "the decorator ``@register_optimizer`` from ``fl_sim.optimizers``"
                    )

    optimizer_cls = get_builtin_optimizer(optimizer_name)
    optimizer = optimizer_cls(params, **_get_cls_init_args(optimizer_cls, config))
    # step_args = inspect.getfullargspec(optimizer.step).args
    # print(f"step_args: {step_args}")
    # if not set(_extra_kwargs).issubset(set(step_args)):
    #     optimizer.step = add_kwargs(
    #         optimizer.step,
    #         **{k: v for k, v in _extra_kwargs.items() if k not in step_args},
    #     )
    #     optimizer.step._with_counter = True
    return optimizer


def _get_cls_init_args(cls: type, config: Any) -> ED:
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
