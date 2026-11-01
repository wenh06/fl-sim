"""
Configs for the server and client nodes.
========================================

"""

import re
import warnings
from pathlib import Path
from typing import Any, ClassVar, List, Literal, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..utils.misc import set_seed

__all__ = [
    "ServerConfig",
    "ClientConfig",
]


class ServerConfig(BaseModel):
    """Configs for the Server.

    Parameters
    ----------
    algorithm : str
        The algorithm name.
        Defaults to be derived from the class name
        (with the ``ServerConfig`` suffix removed).
    num_iters : int
        The number of (outer) iterations.
    num_clients : int, optional
        The number of clients.
        If `None`, will use the dataset's default
        (filled in by :meth:`Server._post_init`).
    clients_sample_ratio : float
        The ratio of clients to sample for each iteration.
    log_dir : str or pathlib.Path, optional
        The log directory.
        If not specified, will use the default log directory.
        If not absolute, will be relative to the default log directory.
    txt_logger : bool, default True
        Whether to use txt logger.
    json_logger : bool, default True
        Whether to use json logger.
    eval_every : int, default 1
        The number of iterations to evaluate the model.
    visiable_gpus : Sequence[int], optional
        Visable GPU IDs for allocating devices for clients.
        Defaults to use all GPUs if available.
    extra_observes : List[str], optional
        Extra attributes to observe during training.
    seed : int, default 0
        The random seed.
    tag : str, optional
        The tag of the experiment.
    verbose : int, default 1
        The verbosity level.
    gpu_proportion : float, default 0.2
        The proportion of clients to use GPU.
        Used to similate the system heterogeneity of the clients.
        Not used in the current version, reserved for future use.
    executor : str, default "sequential"
        The client-executor backend of the parallel execution layer:
        ``"sequential"`` (the default; exactly the pre-parallel behavior)
        or ``"threads"`` (client tasks run on a thread pool).
        Keyword-only: it is a pure execution-layer knob and never enters
        algorithm reproduction configs, so it is excluded from the
        (backward-compatibility) positional-argument mapping.
    num_workers : int, default 0
        The maximum number of client tasks running concurrently
        (threads-backend only); ``0`` means auto (the usable CPU count).
        Keyword-only, like `executor`.
    torch_num_threads : int, optional
        The torch intra-op thread budget set while the threads executor is
        active; ``None`` means auto (``usable cores // num_workers``).
        Keyword-only, like `executor`.
    **kwargs : dict, optional
        The other arguments,
        will be set as attributes of the class.

    .. note::
        Positional arguments are supported for backward compatibility
        (mapped onto the fields in their declaration order),
        but keyword arguments are recommended.

    """

    __name__ = "ServerConfig"

    model_config = ConfigDict(extra="allow")

    # NOTE: the field order matters for (backward-compatible) positional
    # construction, which maps onto the fields in their declaration order;
    # `algorithm` goes last since the (sub)config classes derive it from
    # the class name rather than taking it as the first argument.
    num_iters: int
    num_clients: Optional[int] = None
    clients_sample_ratio: float
    log_dir: Optional[Union[str, Path]] = None
    txt_logger: bool = True
    json_logger: bool = True
    eval_every: int = 1
    visiable_gpus: Optional[List[int]] = None
    extra_observes: List[str] = Field(default_factory=list)
    seed: int = 0
    tag: Optional[str] = None
    verbose: int = 1
    gpu_proportion: float = 0.2
    algorithm: str = ""
    executor: Literal["sequential", "threads"] = "sequential"
    num_workers: int = 0
    torch_num_threads: Optional[int] = None

    # the execution-layer knobs are keyword-only: appending them to the base
    # class would otherwise shift every algorithm subclass's own fields in
    # the (backward-compatibility) positional-argument mapping by the number
    # of appended fields, silently misassigning old positional calls.
    # ClassVar: a plain class attribute, NOT a pydantic (private) field.
    _keyword_only_fields: ClassVar[frozenset] = frozenset({"executor", "num_workers", "torch_num_threads"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # map positional arguments onto the fields in their declaration order,
        # for backward compatibility with the (formerly hand-written) configs
        if args:
            if isinstance(args[0], str):
                # the former (hand-written) configs took the algorithm name as
                # the first positional argument; it is now auto-derived from
                # the class name, and a string in the first positional slot
                # almost surely means an un-migrated old-style call
                raise TypeError(
                    f"{type(self).__name__} received a string ({args[0]!r}) as the first positional argument. "
                    "The former leading `algorithm` positional argument has been removed "
                    "(it is now auto-derived from the class name); "
                    "use keyword arguments, or drop the leading algorithm name."
                )
            fields = [f for f in type(self).model_fields if f not in self._keyword_only_fields]
            if len(args) > len(fields):
                raise TypeError(
                    f"{type(self).__name__} takes at most {len(fields)} positional arguments "
                    f"({len(args)} given); use keyword arguments instead."
                )
            for name, val in zip(fields, args):
                if name in kwargs:
                    raise TypeError(f"{type(self).__name__} got multiple values for argument {name!r}")
                kwargs[name] = val
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def _post_init_config(self) -> "ServerConfig":
        """Derive the algorithm name; filter GPUs; set the global seed."""
        derived = re.sub(r"(Server|Client)Config$", "", type(self).__name__)
        if derived:
            # a conventionally named (sub)config: the algorithm name is fixed to it
            if not self.algorithm:
                self.algorithm = derived
            elif self.algorithm != derived:
                warnings.warn(
                    f"The `algorithm` argument is fixed to `{derived}` for `{type(self).__name__}` " "and will be ignored.",
                    RuntimeWarning,
                )
                self.algorithm = derived
        default_gpus = list(range(torch.cuda.device_count()))
        if self.visiable_gpus is None:
            self.visiable_gpus = default_gpus
        elif not set(self.visiable_gpus).issubset(set(default_gpus)):
            warnings.warn(f"GPU(s) {set(self.visiable_gpus) - set(default_gpus)} " "are not available.")
            self.visiable_gpus = [item for item in self.visiable_gpus if item in default_gpus]
        set_seed(self.seed)
        return self

    def extra_repr_keys(self) -> List[str]:
        """Kept for backward compatibility with the former `ReprMixin`-based configs;
        the fields are all included in the (pydantic) default representation.
        """
        return []


class ClientConfig(BaseModel):
    """Configs for the Client.

    Parameters
    ----------
    algorithm : str
        The algorithm name.
        Defaults to be derived from the class name
        (with the ``ClientConfig`` suffix removed).
    optimizer : str
        The name of the optimizer to solve the local (inner) problem.
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float
        The learning rate.
    scheduler : dict, optional
        The scheduler config.
        None for no scheduler, using constant learning rate.
    extra_observes : List[str], optional
        Extra attributes to observe during training,
        which would be recorded in evaluated metrics,
        sent to the server, and written to the log file.
    verbose : int, default 1
        The verbosity level.
    latency : float, default 0.0
        The latency of the client.
        Not used in the current version, reserved for future use.
    **kwargs : dict, optional
        The other arguments,
        will be set as attributes of the class.

    .. note::
        Positional arguments are supported for backward compatibility
        (mapped onto the fields in their declaration order),
        but keyword arguments are recommended.

    """

    __name__ = "ClientConfig"

    model_config = ConfigDict(extra="allow")

    # NOTE: the field order matters for (backward-compatible) positional
    # construction, which maps onto the fields in their declaration order;
    # `algorithm` goes last since the (sub)config classes derive it from
    # the class name rather than taking it as the first argument.
    batch_size: int
    num_epochs: int
    lr: float
    optimizer: str
    scheduler: Optional[dict] = None
    extra_observes: List[str] = Field(default_factory=list)
    verbose: int = 1
    latency: float = 0.0
    algorithm: str = ""
    rng_mode: Literal["global", "per_client"] = "global"

    # keyword-only knob, see ServerConfig._keyword_only_fields
    _keyword_only_fields: ClassVar[frozenset] = frozenset({"rng_mode"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # map positional arguments onto the fields in their declaration order,
        # for backward compatibility with the (formerly hand-written) configs
        if args:
            if isinstance(args[0], str):
                # the former (hand-written) configs took the algorithm name as
                # the first positional argument; it is now auto-derived from
                # the class name, and a string in the first positional slot
                # almost surely means an un-migrated old-style call
                raise TypeError(
                    f"{type(self).__name__} received a string ({args[0]!r}) as the first positional argument. "
                    "The former leading `algorithm` positional argument has been removed "
                    "(it is now auto-derived from the class name); "
                    "use keyword arguments, or drop the leading algorithm name."
                )
            fields = [f for f in type(self).model_fields if f not in self._keyword_only_fields]
            if len(args) > len(fields):
                raise TypeError(
                    f"{type(self).__name__} takes at most {len(fields)} positional arguments "
                    f"({len(args)} given); use keyword arguments instead."
                )
            for name, val in zip(fields, args):
                if name in kwargs:
                    raise TypeError(f"{type(self).__name__} got multiple values for argument {name!r}")
                kwargs[name] = val
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def _post_init_config(self) -> "ClientConfig":
        """Derive the algorithm name; normalize the scheduler config."""
        derived = re.sub(r"(Server|Client)Config$", "", type(self).__name__)
        if derived:
            # a conventionally named (sub)config: the algorithm name is fixed to it
            if not self.algorithm:
                self.algorithm = derived
            elif self.algorithm != derived:
                warnings.warn(
                    f"The `algorithm` argument is fixed to `{derived}` for `{type(self).__name__}` " "and will be ignored.",
                    RuntimeWarning,
                )
                self.algorithm = derived
        if not self.scheduler:
            self.scheduler = {"name": "none"}
        return self

    def extra_repr_keys(self) -> List[str]:
        """Kept for backward compatibility with the former `ReprMixin`-based configs;
        the fields are all included in the (pydantic) default representation.
        """
        return []
