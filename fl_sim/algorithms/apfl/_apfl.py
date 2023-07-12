"""
"""

import warnings
from typing import List, Dict, Any

import torch  # noqa: F401
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm  # noqa: F401

from ...nodes import (  # noqa: F401
    Server,
    Client,
    ServerConfig,
    ClientConfig,
    ClientMessage,
)  # noqa: F401
from .._register import register_algorithm
from .._misc import server_config_kw_doc, client_config_kw_doc


__all__ = [
    "APFLServerConfig",
    "APFLClientConfig",
    "APFLServer",
    "APFLClient",
]


@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class APFLServerConfig(ServerConfig):
    """Server config for the APFL algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float
        The ratio of clients to sample for each iteration.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "APFLServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        **kwargs: Any,
    ) -> None:
        name = self.__name__.replace("ServerConfig", "")
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                f"The `algorithm` argument is fixed to `{name}` and will be ignored.",
                RuntimeWarning,
            )
        super().__init__(
            name,
            num_iters,
            num_clients,
            clients_sample_ratio,
            **kwargs,
        )


@register_algorithm()
@add_docstring(client_config_kw_doc, "append")
class APFLClientConfig(ClientConfig):
    """Client config for the APFL algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    optimizer : str, default "SGD"
        The name of the optimizer to solve the local (inner) problem.
    lr : float, default 1e-2
        The learning rate.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "APFLClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        optimizer: str = "SGD",
        lr: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        name = self.__name__.replace("ClientConfig", "")
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                f"The `algorithm` argument is fixed to `{name}` and will be ignored.",
                RuntimeWarning,
            )
        super().__init__(
            name,
            optimizer,
            batch_size,
            num_epochs,
            lr,
            **kwargs,
        )


@register_algorithm()
@add_docstring(
    Server.__doc__.replace(
        "The class to simulate the server node.", "Server node for the APFL algorithm."
    )
    .replace("ServerConfig", "APFLServerConfig")
    .replace("ClientConfig", "APFLClientConfig")
)
class APFLServer(Server):
    """Server node for the APFL algorithm."""

    __name__ = "APFLServer"

    @property
    def client_cls(self) -> type:
        return APFLClient

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "APFLClient") -> None:
        target._received_messages = {"parameters": self.get_detached_model_parameters()}

    @add_docstring(Server.update)
    def update(self) -> None:
        raise NotImplementedError

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": APFLServerConfig,
            "client": APFLClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.2003.13461"]


@register_algorithm()
@add_docstring(
    Client.__doc__.replace(
        "The class to simulate the client node.", "Client node for the APFL algorithm."
    ).replace("ClientConfig", "APFLClientConfig")
)
class APFLClient(Client):
    """Client node for the APFL algorithm."""

    __name__ = "APFLClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes for maintaining itermidiate states
        """
        raise NotImplementedError

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "APFLServer") -> None:
        raise NotImplementedError

    def update(self) -> None:
        raise NotImplementedError

    def train(self) -> None:
        """Train (the copy of) the global model with local data."""
        raise NotImplementedError
