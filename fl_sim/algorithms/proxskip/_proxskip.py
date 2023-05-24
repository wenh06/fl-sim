"""
ProxSkip and ProxSkip-VR re-implemented in the new framework
"""

import warnings
from typing import List, Any, Dict

import numpy as np
import torch
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from ...nodes import Server, Client, ServerConfig, ClientConfig, ClientMessage
from ...data_processing.fed_dataset import FedDataset


__all__ = [
    "ProxSkipServer",
    "ProxSkipClient",
    "ProxSkipServerConfig",
    "ProxSkipClientConfig",
]


class ProxSkipServerConfig(ServerConfig):
    """Server config for the ProxSkip algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    p : float
        Probability of skipping communication.
    vr : bool, default False
        Whether to use variance reduction.
    **kwargs : dict, optional
        Additional keyword arguments:

        - ``txt_logger`` : bool, default True
            Whether to use txt logger.
        - ``csv_logger`` : bool, default False
            Whether to use csv logger.
        - ``json_logger`` : bool, default True
            Whether to use json logger.
        - ``eval_every`` : int, default 1
            The number of iterations to evaluate the model.
        - ``verbose`` : int, default 1
            The verbosity level.

    """

    __name__ = "ProxSkipServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        p: float,
        vr: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "ProxSkip",
            num_iters,
            num_clients,
            clients_sample_ratio=1,
            p=p,
            vr=vr,
            **kwargs,
        )


class ProxSkipClientConfig(ClientConfig):
    """Client config for the ProxSkip algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 1e-2
        The learning rate.
    vr : bool, default False
        Whether to use variance reduction.
    **kwargs : dict, optional
        Additional keyword arguments:

        - ``verbose`` : int, default 1
            The verbosity level.

    """

    __name__ = "ProxSkipClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        vr: bool = False,
        **kwargs: Any,
    ) -> None:
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                "The `algorithm` argument fixed to `ProxSkip`.", RuntimeWarning
            )
        if kwargs.pop("optimizer", None) is not None:
            warnings.warn(
                "The `optimizer` argument fixed to `SCAFFOLD`.", RuntimeWarning
            )
        super().__init__(
            "ProxSkip",
            "SCAFFOLD",
            batch_size,
            num_epochs,
            lr,
            vr=vr,
            **kwargs,
        )


@add_docstring(
    Server.__doc__.replace(
        "The class to simulate the server node.",
        "Server node for the ProxSkip algorithm.",
    )
    .replace("ServerConfig", "ProxSkipServerConfig")
    .replace("ClientConfig", "ProxSkipClientConfig")
)
class ProxSkipServer(Server):
    """Server node for the ProxSkip algorithm."""

    __name__ = "ProxSkipServer"

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: FedDataset,
        config: ProxSkipServerConfig,
        client_config: ProxSkipClientConfig,
    ) -> None:

        # assign communication pattern to client config
        setattr(client_config, "p", config.p)
        super().__init__(model, dataset, config, client_config)

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and check compatibility of server and client configs
        """
        super()._post_init()
        assert self.config.vr == self._client_config.vr
        # ProxSkip does not have control variates on the server side
        # self._control_variates = [torch.zeros_like(p) for p in self.model.parameters()]
        communication_pattern = (
            np.random.rand(self.config.num_iters) < self.config.p
        ).astype(int)
        for c in self._clients:
            c.communication_pattern = communication_pattern

    @property
    def client_cls(self) -> type:
        return ProxSkipClient

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "ProxSkipClient") -> None:
        target._received_messages = {
            "parameters": [p.detach().clone() for p in self.model.parameters()],
        }
        if target.config.vr:
            target._received_messages["gradients"] = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in target.model.parameters()
            ]

    def update(self) -> None:

        # sum of received parameters, with self.model.parameters() as its container
        self.avg_parameters()
        if self.config.vr:
            self.update_gradients()

    def aggregate_client_metrics(self) -> None:
        """skip aggregation if no client has communicated"""
        if len(self._received_messages) == 0:
            return
        super().aggregate_client_metrics()

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": ProxSkipServerConfig,
            "client": ProxSkipClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.2202.09357"]


@add_docstring(
    Client.__doc__.replace(
        "The class to simulate the client node.",
        "Client node for the ProxSkip algorithm.",
    ).replace("ClientConfig", "ProxSkipClientConfig")
)
class ProxSkipClient(Client):
    """Client node for the ProxSkip algorithm."""

    __name__ = "ProxSkipClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes for maintaining itermidiate states
        """
        super()._post_init()
        self._control_variates = [torch.zeros_like(p) for p in self.model.parameters()]
        if self.config.vr:
            self._gradient_buffer = [
                torch.zeros_like(p) for p in self.model.parameters()
            ]
        else:
            self._gradient_buffer = None
        self.communication_pattern = (
            None  # would be set by the server in its `_post_init` method
        )

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "ProxSkipServer") -> None:
        if self.communication_pattern[target.n_iter] == 0:
            return  # skip communication
        message = {
            "client_id": self.client_id,
            "parameters": self.get_detached_model_parameters(),
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        if self.config.vr:
            message["gradients"] = [
                p.grad.detach().clone() for p in self.model.parameters()
            ]
        target._received_messages.append(ClientMessage(**message))

    def update(self) -> None:
        try:
            self.set_parameters(self._received_messages["parameters"])
        except KeyError:
            warnings.warn(
                "No parameters received from server. "
                "Using current model parameters as initial parameters.",
                RuntimeWarning,
            )
        except Exception as err:
            raise err
        if (
            self.config.vr
            and self._received_messages.get("gradients", None) is not None
        ):
            self._gradient_buffer = [
                gd.clone().to(self.device)
                for gd in self._received_messages["gradients"]
            ]

        # update the control variates (indeed the last step of the previous local iteration)
        if self._received_messages.get("parameters", None) is not None:
            for mp, rp, cv in zip(
                self.model.parameters(),
                self._received_messages["parameters"],
                self._control_variates,
            ):
                cv.add_(
                    mp.data - rp.data.to(self.device),
                    alpha=self.config.p / self.config.lr,
                )

        self.solve_inner()  # alias of self.train()

    def train(self) -> None:
        self.model.train()
        with tqdm(
            range(self.config.num_epochs),
            total=self.config.num_epochs,
            mininterval=1.0,
            disable=self.config.verbose < 2,
        ) as pbar:
            for epoch in pbar:  # local update
                self.model.train()
                for X, y in self.train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(X)
                    loss = self.criterion(output, y)
                    loss.backward()
                    variance_buffer = [cv.clone() for cv in self._control_variates]
                    if self._gradient_buffer is not None:
                        for cv, gd in zip(variance_buffer, self._gradient_buffer):
                            cv.add_(gd.clone())
                    self.optimizer.step(
                        variance_buffer=variance_buffer,
                    )
                    # free memory
                    del loss, output, X, y
        # control variates is updated after communication before solve_inner in the next iteration
