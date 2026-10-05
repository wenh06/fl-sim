"""
Fusion Penalized Federated Clustering (FPFC) algorithm.

`Clustered Federated Learning based on Nonconvex Pairwise Fusion. <https://arxiv.org/abs/2211.04218>`_

Codebase URL: https://github.com/RUC-GSAI/Yu-etal-2024-FPFC

"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List

import torch
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from ...nodes import ClientMessage
from .._misc import client_config_kw_doc, server_config_kw_doc
from .._register import register_algorithm
from ..fedopt import FedAvgClient as BaseClient
from ..fedopt import FedAvgClientConfig as BaseClientConfig
from ..fedopt import FedAvgServer as BaseServer
from ..fedopt import FedAvgServerConfig as BaseServerConfig

__all__ = [
    "FPFCClient",
    "FPFCClientConfig",
    "FPFCServer",
    "FPFCServerConfig",
]


_base_algorithm = "FedAvg"


@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class FPFCServerConfig(BaseServerConfig):
    """Server config for the FPFC algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float, default 1
        The ratio of clients to participate in each round.
    rho : float, default 1.0
        The penalty parameter :math:`\\rho` of the augmented Lagrangian (ADMM).
    xi : float, default 1e-4
        The smoothing parameter :math:`\\xi` of the (smoothed SCAD-type) fusion penalty.
    lam : float, default 1.0
        The penalty coefficient :math:`\\lambda` of the pairwise fusion penalty.
    a : float, default 3.7
        The parameter :math:`a` of the fusion penalty, should be larger than 1.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FPFCServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float = 1,
        rho: float = 1.0,
        xi: float = 1e-4,
        lam: float = 1.0,
        a: float = 3.7,
        **kwargs: Any,
    ) -> None:
        name = self.__name__.replace("ServerConfig", "")
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                f"The `algorithm` argument is fixed to `{name}` and will be ignored.",
                RuntimeWarning,
            )
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio=clients_sample_ratio,
            **kwargs,
        )
        self.algorithm = name
        self.rho = rho
        self.xi = xi
        self.lam = lam
        self.a = a


@register_algorithm()
@add_docstring(client_config_kw_doc, "append")
class FPFCClientConfig(BaseClientConfig):
    """Client config for the FPFC algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of (local) epochs, i.e. :math:`T_i` in the paper.
    lr : float, default 1e-2
        The learning rate, i.e. the stepsize :math:`\\alpha` in the paper.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FPFCClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
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
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            **kwargs,
        )
        self.algorithm = name


@register_algorithm()
@add_docstring(BaseServer.__doc__.replace(_base_algorithm, "FPFC"))
class FPFCServer(BaseServer):
    __name__ = "FPFCServer"

    def _post_init(self) -> None:
        """Check the configs and initialize the auxiliary sequences."""
        super()._post_init()
        rho, xi, lam, a = self.config.rho, self.config.xi, self.config.lam, self.config.a
        assert rho > 0, f"`rho` should be positive, but got {rho}."
        assert xi > 0, f"`xi` should be positive, but got {xi}."
        assert lam > 0, f"`lam` should be positive, but got {lam}."
        assert a > 1, f"`a` should be larger than 1, but got {a}."
        assert abs((a - 1) * rho - 1) > 1e-12, "`(a - 1) * rho` should not be 1."
        # cache of each client's latest model parameters, i.e. {omega_i}
        self._model_params = {
            client_id: [p.detach().clone() for p in self.model.parameters()] for client_id in range(self.config.num_clients)
        }
        # the (fused) pairwise difference variables {theta_ij}, stored sparsely
        # for pairs (i, j) (i < j) that have been updated at least once
        self._theta: Dict[int, Dict[int, List[torch.Tensor]]] = {i: {} for i in range(self.config.num_clients)}
        # the dual variables {v_ij}, stored sparsely, aligned with `self._theta`
        self._v: Dict[int, Dict[int, List[torch.Tensor]]] = {i: {} for i in range(self.config.num_clients)}
        # the auxiliary variables {zeta_i} broadcast to the clients
        self._zeta = deepcopy(self._model_params)

    @property
    def client_cls(self) -> type:
        return FPFCClient

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FPFCServerConfig,
            "client": FPFCClientConfig,
        }

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def _fusion_prox(self, delta: List[torch.Tensor]) -> List[torch.Tensor]:
        """The proximal operator of the (scaled) pairwise fusion penalty,
        i.e. Eq. (6) of the paper.

        Parameters
        ----------
        delta : List[torch.Tensor]
            The (list of) pairwise difference(s) :math:`\\delta_{ij} = \\omega_i - \\omega_j + v_{ij} / \\rho`.

        Returns
        -------
        List[torch.Tensor]
            The updated pairwise difference variables :math:`\\theta_{ij}`.

        """
        rho, xi, lam, a = self.config.rho, self.config.xi, self.config.lam, self.config.a
        norm = torch.sqrt(sum((d**2).sum() for d in delta))
        if norm <= xi + lam / rho:
            coef = xi * rho / (lam + xi * rho)
        elif norm <= lam + lam / rho:
            coef = 1 - lam / (rho * norm)
        elif norm <= a * lam:
            coef = max(0.0, 1 - a * lam / ((a - 1) * rho * norm)) / (1 - 1 / ((a - 1) * rho))
        else:
            coef = 1.0
        return [coef * d for d in delta]

    def communicate(self, target: "FPFCClient") -> None:
        """Send the auxiliary variable zeta and the penalty parameter rho to the client."""
        target._received_messages = {
            "parameters": deepcopy(self._zeta[target.client_id]),
            "rho": self.config.rho,
        }

    @torch.no_grad()
    def update(self) -> None:
        """Update the pairwise fusion variables, the dual variables, and the auxiliary variables."""
        active_ids = []
        for m in self._received_messages:
            client_id = m["client_id"]
            self._model_params[client_id] = [p.detach().clone().to(self.device) for p in m["parameters"]]
            active_ids.append(client_id)
        rho = self.config.rho
        # update theta_ij and v_ij for pairs (i, j) (i < j) of active devices
        for ii, i in enumerate(active_ids):
            for j in active_ids[ii + 1 :]:
                if j in self._theta[i]:
                    # (i, j) with i < j stored in self._theta[i]
                    theta_ij, v_ij = self._theta[i][j], self._v[i][j]
                elif i in self._theta[j]:
                    # (j, i) with j < i stored in self._theta[j]
                    theta_ij = [-t for t in self._theta[j][i]]
                    v_ij = [-v for v in self._v[j][i]]
                else:  # unseen pair, initialize with zeros
                    theta_ij = [torch.zeros_like(p) for p in self._model_params[i]]
                    v_ij = [torch.zeros_like(p) for p in self._model_params[i]]
                delta = [wi - wj + v / rho for wi, wj, v in zip(self._model_params[i], self._model_params[j], v_ij)]
                new_theta = self._fusion_prox(delta)
                new_v = [v + rho * (dk - tk) for v, dk, tk in zip(v_ij, delta, new_theta)]
                if i < j:
                    self._theta[i][j], self._v[i][j] = new_theta, new_v
                else:
                    self._theta[j][i], self._v[j][i] = [-t for t in new_theta], [-v for v in new_v]
        # update zeta_i for all clients i in [m]:
        # zeta_i = (1/m) sum_j (omega_j + theta_ij - v_ij / rho)
        mean_omega = [torch.zeros_like(p) for p in self.model.parameters()]
        for params in self._model_params.values():
            for mp, p in zip(mean_omega, params):
                mp += p
        mean_omega = [mp / self.config.num_clients for mp in mean_omega]
        for i in range(self.config.num_clients):
            corr = [torch.zeros_like(p) for p in mean_omega]
            for j in range(i + 1, self.config.num_clients):  # pairs (i, j) with i < j
                if j in self._theta[i]:
                    for c, t, v in zip(corr, self._theta[i][j], self._v[i][j]):
                        c += t - v / rho
            for j in range(i):  # pairs (j, i) with j < i
                if i in self._theta[j]:
                    for c, t, v in zip(corr, self._theta[j][i], self._v[j][i]):
                        c -= t - v / rho
            self._zeta[i] = [z + c / self.config.num_clients for z, c in zip(mean_omega, corr)]
        # the server model is set to the average of all (personalized) models,
        # merely for centralized evaluation and logging
        self.set_parameters(mean_omega)


@register_algorithm()
@add_docstring(BaseClient.__doc__.replace(_base_algorithm, "FPFC"))
class FPFCClient(BaseClient):
    __name__ = "FPFCClient"

    def _post_init(self) -> None:
        """Check if all required fields in the config are set,
        and set attributes for maintaining intermediate states.
        """
        super()._post_init()
        self._rho = 1.0

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "FPFCServer") -> None:
        message = {
            "client_id": self.client_id,
            "parameters": self.get_detached_model_parameters(),
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        target._received_messages.append(ClientMessage(**message))

    def update(self) -> None:
        try:
            self._cached_parameters = deepcopy(self._received_messages["parameters"])
            self._rho = self._received_messages.get("rho", self._rho)
        except KeyError:
            warnings.warn(
                "No parameters received from server. " "Using current model parameters as initial parameters.",
                RuntimeWarning,
            )
            self._cached_parameters = self.get_detached_model_parameters()
        except Exception as err:
            raise err
        self._cached_parameters = [p.to(self.device) for p in self._cached_parameters]
        self.set_parameters(self._cached_parameters)
        self.solve_inner()  # alias of self.train()

    def train(self) -> None:
        """Local (stochastic) gradient descent on the objective
        :math:`f_i(\\omega) + \\rho / 2 \\lVert \\omega - \\zeta_i \\rVert^2`,
        i.e. Eq. (5) of the paper.
        """
        self.model.train()
        with torch.no_grad():
            for p, z in zip(self.model.parameters(), self._cached_parameters):
                p.data.copy_(z.data.to(p.device))
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
                    # the penalty term rho / 2 * ||omega - zeta_i||^2
                    penalty = (
                        self._rho
                        / 2
                        * sum(((p - z) ** 2).sum() for p, z in zip(self.model.parameters(), self._cached_parameters))
                    )
                    loss = loss + penalty
                    loss.backward()
                    self.optimizer.step()
                    # free memory
                    # del X, y, output, loss
        self.lr_scheduler.step()
