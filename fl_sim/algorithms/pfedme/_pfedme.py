"""
pFedMe re-implemented in the new framework
"""

import warnings
from copy import deepcopy
from typing import List, Any, Dict

from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from ...nodes import (
    Server,
    Client,
    ServerConfig,
    ClientConfig,
    ClientMessage,
)


__all__ = [
    "pFedMeServer",
    "pFedMeClient",
    "pFedMeServerConfig",
    "pFedMeClientConfig",
]


class pFedMeServerConfig(ServerConfig):
    """Server config for the pFedMe algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float
        The ratio of clients to sample for each iteration.
    beta : float, default 1.0
        The beta (inertia) parameter for model aggregation.
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
        - ``seed`` : int, default 0
            The random seed.
        - ``verbose`` : int, default 1
            The verbosity level.

    """

    __name__ = "pFedMeServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        beta: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "pFedMe", num_iters, num_clients, clients_sample_ratio, beta=beta, **kwargs
        )


class pFedMeClientConfig(ClientConfig):
    """Client config for the pFedMe algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 5e-3
        The learning rate for personalized model training.
    num_steps : int, default 30
        The number of steps for each epoch.
    lamda : float, default 15.0
        The lambda parameter for pFedMe,
        i.e. the coefficient of the proximal term.
    eta : float, default 1e-3
        The eta (learning rate) parameter for pFedMe.
    mu : float, default 1e-3
        The mu (momentum) parameter for pFedMe.
    **kwargs : dict, optional
        Additional keyword arguments:

        - ``verbose`` : int, default 1
            The verbosity level.

    References
    ----------
    1. https://github.com/CharlieDinh/pFedMe/blob/master/FLAlgorithms/users/userpFedMe.py

    Note:
    1. `lr` is the `personal_learning_rate` in the original implementation
    2. `eta` is the `learning_rate` in the original implementation
    3. `mu` is the momentum factor in the original implemented optimzer
    """

    __name__ = "pFedMeClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 5e-3,
        num_steps: int = 30,
        lamda: float = 15.0,
        eta: float = 1e-3,
        mu: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn("The `algorithm` argument fixed to `pFedMe`.", RuntimeWarning)
        if kwargs.pop("optimizer", None) is not None:
            warnings.warn("The `optimizer` argument fixed to `pFedMe`.", RuntimeWarning)
        super().__init__(
            "pFedMe",
            "pFedMe",
            batch_size,
            num_epochs,
            lr,
            num_steps=num_steps,
            lamda=lamda,
            eta=eta,
            mu=mu,
            **kwargs,
        )


@add_docstring(
    Server.__doc__.replace(
        "The class to simulate the server node.",
        "Server node for the pFedMe algorithm.",
    )
    .replace("ServerConfig", "pFedMeServerConfig")
    .replace("ClientConfig", "pFedMeClientConfig")
)
class pFedMeServer(Server):
    """Server node for the pFedMe algorithm."""

    __name__ = "pFedMeServer"

    @property
    def client_cls(self) -> type:
        return pFedMeClient

    @property
    def required_config_fields(self) -> List[str]:
        return ["beta"]

    def communicate(self, target: "pFedMeClient") -> None:
        target._received_messages = {"parameters": self.get_detached_model_parameters()}

    def update(self) -> None:

        # store previous parameters
        previous_params = self.get_detached_model_parameters()
        for p in previous_params:
            p = p.to(self.device)

        # sum of received parameters, with self.model.parameters() as its container
        self.avg_parameters()

        # aaggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(previous_params, self.model.parameters()):
            param.data = (
                1 - self.config.beta
            ) * pre_param.data.detach().clone() + self.config.beta * param.data

        # clear received messages
        del pre_param

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": pFedMeServerConfig,
            "client": pFedMeClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.2006.08848"]


@add_docstring(
    Client.__doc__.replace(
        "The class to simulate the client node.",
        "Client node for the pFedMe algorithm.",
    ).replace("ClientConfig", "pFedMeClientConfig")
)
class pFedMeClient(Client):
    """Client node for the pFedMe algorithm."""

    __name__ = "pFedMeClient"

    @property
    def required_config_fields(self) -> List[str]:
        return ["num_steps", "lamda", "eta", "mu"]

    def communicate(self, target: "pFedMeServer") -> None:
        target._received_messages.append(
            ClientMessage(
                **{
                    "client_id": self.client_id,
                    "parameters": self.get_detached_model_parameters(),
                    "train_samples": self.config.num_epochs * self.config.batch_size,
                    "metrics": self._metrics,
                }
            )
        )

    def update(self) -> None:

        # copy the parameters from the server
        # pFedMe paper Algorithm 1 line 5
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
        # update the model via prox_sgd
        # pFedMe paper Algorithm 1 line 6 - 8
        self.solve_inner()  # alias of self.train()

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
                X, y = self.sample_data()
                X, y = X.to(self.device), y.to(self.device)
                # personalized steps
                for i in range(self.config.num_steps):
                    self.optimizer.zero_grad()
                    output = self.model(X)
                    loss = self.criterion(output, y)
                    loss.backward()
                    self.optimizer.step(self._cached_parameters)

                # update local weight after finding aproximate theta
                # pFedMe paper Algorithm 1 line 8
                for mp, cp in zip(self.model.parameters(), self._cached_parameters):
                    # print(mp.data.isnan().any(), cp.data.isnan().any())
                    cp.data.add_(
                        cp.data.clone() - mp.data.clone(),
                        alpha=-self.config.lamda * self.config.eta,
                    )

                # update local model
                # the init parameters (theta in pFedMe paper Algorithm 1 line  7) for the next iteration
                # are set to be `self._cached_parameters`
                self.set_parameters(self._cached_parameters)

                # free memory
                del X, y, output, loss
