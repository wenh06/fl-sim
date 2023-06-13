"""
"""

import warnings
from copy import deepcopy
from typing import List, Dict, Any, Optional

from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from ...nodes import Server, Client, ServerConfig, ClientConfig, ClientMessage
from ...data_processing.fed_dataset import FedDataset
from .._register import _register_algorithm


__all__ = [
    "FedSplitServer",
    "FedSplitClient",
    "FedSplitServerConfig",
    "FedSplitClientConfig",
]


@_register_algorithm("FedSplit")
class FedSplitServerConfig(ServerConfig):
    """Server config for the FedSplit algorithm.

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

        - ``log_dir`` : str or Path, optional
            The log directory.
            If not specified, will use the default log directory.
            If not absolute, will be relative to the default log directory.
        - ``txt_logger`` : bool, default True
            Whether to use txt logger.
        - ``json_logger`` : bool, default True
            Whether to use json logger.
        - ``eval_every`` : int, default 1
            The number of iterations to evaluate the model.
        - ``visiable_gpus`` : Sequence[int], optional
            Visable GPU IDs for allocating devices for clients.
            Defaults to use all GPUs if available.
        - ``seed`` : int, default 0
            The random seed.
        - ``tag`` : str, optional
            The tag of the experiment.
        - ``verbose`` : int, default 1
            The verbosity level.
        - ``gpu_proportion`` : float, default 0.2
            The proportion of clients to use GPU.
            Used to similate the system heterogeneity of the clients.
            Not used in the current version, reserved for future use.

    """

    __name__ = "FedSplitServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "FedSplit",
            num_iters,
            num_clients,
            clients_sample_ratio,
            **kwargs,
        )


@_register_algorithm("FedSplit")
class FedSplitClientConfig(ClientConfig):
    """Client config for the FedSplit algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 1e-2
        The learning rate.
    s : float, default 10.0
        Reciprocal of the proximal parameter.
    **kwargs : dict, optional
        Additional keyword arguments:

        - ``scheduler`` : dict, optional
            The scheduler config.
            None for no scheduler, using constant learning rate.
        - ``verbose`` : int, default 1
            The verbosity level.
        - ``latency`` : float, default 0.0
            The latency of the client.
            Not used in the current version, reserved for future use.

    """

    __name__ = "FedSplitClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        s: float = 10.0,
        **kwargs: Any,
    ) -> None:
        self.s = s
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                "The `algorithm` argument fixed to `FedSplit`.", RuntimeWarning
            )
        if kwargs.pop("optimizer", None) is not None:
            warnings.warn(
                "The `optimizer` argument fixed to `ProxSGD`.", RuntimeWarning
            )
        super().__init__(
            "FedSplit",
            "ProxSGD",
            batch_size,
            num_epochs,
            lr,
            prox=1.0 / s,
            **kwargs,
        )


@_register_algorithm("FedSplit")
@add_docstring(
    Server.__doc__.replace(
        "The class to simulate the server node.",
        "Server node for the FedSplit algorithm.",
    )
    .replace("ServerConfig", "FedSplitServerConfig")
    .replace("ClientConfig", "FedSplitClientConfig")
)
class FedSplitServer(Server):
    """Server node for the FedSplit algorithm."""

    __name__ = "FedSplitServer"

    def _setup_clients(
        self,
        dataset: Optional[FedDataset] = None,
        client_config: Optional[ClientConfig] = None,
        force: bool = False,
    ) -> None:
        """Setup the clients.

        Parameters
        ----------
        dataset : FedDataset, optional
            The dataset to be used for training the local models,
            defaults to `self.dataset`.
        client_config : ClientConfig, optional
            The configs for the clients,
            defaults to `self._client_config`.
        force : bool, default False
            Whether to force setup the clients.
            If set to True, the clients will be setup
            even if they have been setup before.

        Returns
        -------
        None

        """
        super()._setup_clients(dataset, client_config, force)
        for c in self._clients:
            # line 2 of Algorithm 1 in the paper
            c._z_parameters = [
                p.to(c.device) for p in self.get_detached_model_parameters()
            ]

    @property
    def required_config_fields(self) -> List[str]:
        return []

    @property
    def client_cls(self) -> type:
        return FedSplitClient

    def communicate(self, target: "FedSplitClient") -> None:
        target._received_messages = {"parameters": self.get_detached_model_parameters()}

    def update(self) -> None:
        self.avg_parameters(size_aware=False)  # line 8 of Algorithm 1 in the paper

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FedSplitServerConfig,
            "client": FedSplitClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.2005.05238"]


@_register_algorithm("FedSplit")
@add_docstring(
    Client.__doc__.replace(
        "The class to simulate the client node.",
        "Client node for the FedSplit algorithm.",
    ).replace("ClientConfig", "FedSplitClientConfig")
)
class FedSplitClient(Client):
    """Client node for the FedSplit algorithm."""

    __name__ = "FedSplitClient"

    def _post_init(self) -> None:
        super()._post_init()
        self._z_parameters = None
        # self._z_half_parameters = None  # self.model.parameters() holds this

    @property
    def required_config_fields(self) -> List[str]:
        return ["s"]

    def communicate(self, target: "FedSplitServer") -> None:
        target._received_messages.append(
            ClientMessage(
                **{
                    "client_id": self.client_id,
                    "parameters": [p.clone() for p in self._z_parameters],
                    "train_samples": len(self.train_loader.dataset),
                    "metrics": self._metrics,
                }
            )
        )

    def update(self) -> None:
        try:
            self._cached_parameters = deepcopy(self._received_messages["parameters"])
        except KeyError:
            warnings.warn(
                "No parameters received from server. "
                "Using current model parameters as initial parameters.",
                RuntimeWarning,
            )
            self._cached_parameters = self.get_detached_model_parameters()
        except Exception as err:
            raise err
        self._cached_parameters = [p.to(self.device) for p in self._cached_parameters]
        # Local prox step: line 5 of Algorithm 1 in the paper
        self.solve_inner()  # alias of self.train()
        # Local centering step: line 6 of Algorithm 1 in the paper
        for (zp, mp, cp) in zip(
            self._z_parameters, self.model.parameters(), self._cached_parameters
        ):
            zp.add_(mp.detach().clone().sub(cp.detach().clone()), alpha=2.0)

    def train(self) -> None:
        self.model.train()
        with tqdm(
            range(self.config.num_epochs),
            total=self.config.num_epochs,
            mininterval=1.0,
            disable=self.config.verbose < 2,
            leave=False,
        ) as pbar:
            for epoch in pbar:  # local update
                self.model.train()
                for X, y in self.train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(X)
                    loss = self.criterion(output, y)
                    loss.backward()
                    self.optimizer.step(
                        local_weights=[
                            (2.0 * cp.detach().clone()).sub(zp.detach().clone())
                            for (cp, zp) in zip(
                                self._cached_parameters, self._z_parameters
                            )
                        ]
                    )
                    # free memory
                    del X, y, output, loss
        self.lr_scheduler.step()
