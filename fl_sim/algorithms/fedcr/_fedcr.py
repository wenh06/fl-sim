"""
FedCR: Personalized Federated Learning Based on Across-Client Common Representation
with Conditional Mutual Information Regularization (ICML 2023).

Implementation notes
--------------------
The model is split into a **feature extractor** (all parameters except the last
``nn.Linear`` classifier) and a **personal classifier head** (the last ``nn.Linear``):
the extractor is aggregated at the server while the head stays local (as required
by the algorithm). The exchange uses the (recommended) named-parameter format
(``model_state``, cf. :data:`fl_sim.messages.ModelState`) restricted to the
extractor's parameters, so no positional index bookkeeping is needed. The
global/common class-wise feature representation is tracked
as diagonal Gaussians aggregated via product-of-experts (PoE). The local objective
adds a KL alignment term between the local class-wise feature distribution and the
global one (the conditional-mutual-information regularizer). This implementation
uses deterministic features (``\\Sigma(x) = 0``), which the paper discusses as the
degenerate case that reduces the stochastic network to FedPer plus the
class-wise feature alignment.

"""

from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from ...messages import ClientToServer, ServerToClient
from .._misc import client_config_kw_doc, server_config_kw_doc
from .._register import register_algorithm
from ..fedopt import FedAvgClient as BaseClient
from ..fedopt import FedAvgClientConfig as BaseClientConfig
from ..fedopt import FedAvgServer as BaseServer
from ..fedopt import FedAvgServerConfig as BaseServerConfig

__all__ = [
    "FedCRClient",
    "FedCRClientConfig",
    "FedCRServer",
    "FedCRServerConfig",
]


_base_algorithm = "FedAvg"


def _classifier_name(model: torch.nn.Module) -> str:
    """Return the name of the last ``nn.Linear`` module of the model,
    which is treated as the personal classifier head."""
    name = None
    for n, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            name = n
    if name is None:
        raise ValueError(f"No `nn.Linear` classifier found in the model `{type(model).__name__}`.")
    return name


def _is_head(name: str, head_name: str) -> bool:
    return name == head_name or name.startswith(head_name + ".")


@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class FedCRServerConfig(BaseServerConfig):
    """Server config for the FedCR algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float, default 1
        The ratio of clients to participate in each round.
    beta : float, default 0.005
        The coefficient :math:`\\beta` of the CMI (KL alignment) regularizer.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FedCRServerConfig"

    clients_sample_ratio: float = 1
    beta: float = 0.005


@register_algorithm()
@add_docstring(client_config_kw_doc, "append")
class FedCRClientConfig(BaseClientConfig):
    """Client config for the FedCR algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of (local) epochs.
    lr : float, default 1e-2
        The learning rate.
    **kwargs : dict, optional
        Additional keyword arguments:
    """

    __name__ = "FedCRClientConfig"

    lr: float = 1e-2


@register_algorithm()
@add_docstring(BaseServer.__doc__.replace(_base_algorithm, "FedCR"))
class FedCRServer(BaseServer):
    __name__ = "FedCRServer"

    def _post_init(self) -> None:
        """Check the configs and initialize the global common representation."""
        super()._post_init()
        assert self.config.beta >= 0, f"`beta` should be non-negative, but got {self.config.beta}."
        self._head_name = _classifier_name(self.model)
        # global class-wise common representation: {class_id: (mu, var)}, diagonal Gaussians
        self._global_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def client_cls(self) -> type:
        return FedCRClient

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FedCRServerConfig,
            "client": FedCRClientConfig,
        }

    def _encoder_state(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """The feature extractor state as a named tensor mapping,
        i.e. all model parameters except the personal classifier head,
        in the (recommended) :data:`~fl_sim.messages.ModelState` format."""
        return {n: p.detach().clone() for n, p in model.named_parameters() if not _is_head(n, self._head_name)}

    def _set_encoder(self, model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
        """Load the (aggregated) feature extractor state into the model, by name."""
        for n, p in model.named_parameters():
            if _is_head(n, self._head_name):
                continue
            assert n in state, (
                "the received global feature extractor misses the parameter "
                f"`{n}`; the encoder/head split of the two sides does not match"
            )
            p.data.copy_(state[n].data.to(p.device))

    def communicate(self, target: "FedCRClient") -> None:
        # named exchange: only the feature extractor leaves the server;
        # the global common representation and the regularizer coefficient go as extras
        target.receive(
            ServerToClient(
                model_state=deepcopy(self._encoder_state(self.model)),
                extras={
                    "global_stats": deepcopy(self._global_stats),
                    "beta": self.config.beta,
                },
            )
        )

    @torch.no_grad()
    def update(self) -> None:
        """Aggregate the feature extractors, and aggregate the class-wise
        local feature distributions into the global common representation via PoE
        (with the standard normal prior :math:`p(z) \\sim \\mathcal{N}(0, 1)`),
        cf. Eq. (8) and Eq. (9) of the paper.
        """
        # aggregate the feature extractors (the personal heads are kept local)
        total_samples = sum(m["train_samples"] for m in self._received_messages)
        aggregated = {n: torch.zeros_like(p) for n, p in self._encoder_state(self.model).items()}
        for m in self._received_messages:
            ratio = m["train_samples"] / total_samples
            for n, v in m["model_state"].items():
                aggregated[n] += ratio * v.detach().clone().to(self.device)
        self._set_encoder(self.model, aggregated)
        # aggregate the class-wise feature distributions via PoE:
        # precision = prior precision (1) + sum_i (count_i / var_i)
        # mu = precision^{-1} * sum_i (count_i * mu_i / var_i)
        collected: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]] = {}
        for m in self._received_messages:
            for class_id, (mu, var, count) in m["class_stats"].items():
                collected.setdefault(class_id, []).append(
                    (
                        mu.detach().clone().to(self.device),
                        var.detach().clone().to(self.device),
                        int(count),
                    )
                )
        for class_id, entries in collected.items():
            mu_acc = None
            prec = None
            for mu, var, count in entries:
                p = count / var
                prec = p if prec is None else prec + p
                mu_acc = count * mu / var if mu_acc is None else mu_acc + count * mu / var
            # the standard normal prior p(z) ~ N(0, 1) contributes precision 1 and mean 0
            prec = prec + torch.ones_like(prec)
            var = 1 / prec
            mu = var * mu_acc
            self._global_stats[class_id] = (mu, var)


@register_algorithm()
@add_docstring(BaseClient.__doc__.replace(_base_algorithm, "FedCR"))
class FedCRClient(BaseClient):
    __name__ = "FedCRClient"

    def _post_init(self) -> None:
        """Check if all required fields in the config are set,
        and set attributes for maintaining intermediate states.
        """
        super()._post_init()
        self._head_name = _classifier_name(self.model)
        self._global_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # round-level accumulators of the local class-wise feature statistics: {class_id: [sum, sum_sq, count]}
        self._local_stats: Dict[int, List[torch.Tensor]] = {}
        self._features_buffer: torch.Tensor = None
        # the coefficient of the CMI (KL alignment) regularizer, sent by the server
        self._beta = 0.005
        # capture the input of the classifier head (the extracted features) via a forward pre-hook
        head_module = dict(self.model.named_modules())[self._head_name]
        head_module.register_forward_pre_hook(self._capture_features)

    def _capture_features(self, module: torch.nn.Module, args: tuple) -> None:
        self._features_buffer = args[0].detach()

    def _encoder_state(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """The feature extractor state as a named tensor mapping,
        i.e. all model parameters except the personal classifier head,
        in the (recommended) :data:`~fl_sim.messages.ModelState` format."""
        return {n: p.detach().clone() for n, p in model.named_parameters() if not _is_head(n, self._head_name)}

    def accept_parameters(self) -> None:
        """Accept only the (aggregated) feature extractor parameters;
        the personal classifier head never leaves the client."""
        state = self._received_messages["model_state"]
        for n, p in self.model.named_parameters():
            if _is_head(n, self._head_name):
                continue
            assert n in state, (
                "the received global feature extractor misses the parameter "
                f"`{n}`; the encoder/head split of the two sides does not match"
            )
            p.data.copy_(state[n].data.to(p.device))

    def communicate(self) -> ClientToServer:
        # named exchange: upload the feature extractor state (the personal head never leaves);
        # the round-level class-wise local feature statistics go as extras:
        # mu = sum / count, var = sum_sq / count - mu^2
        class_stats = {}
        for class_id, (sum_, sum_sq, count) in self._local_stats.items():
            mu = sum_ / count
            var = sum_sq / count - mu**2
            class_stats[class_id] = (mu, var.clamp_min(1e-6), count)
        return ClientToServer(
            client_id=self.client_id,
            train_samples=len(self.train_loader.dataset),
            metrics=self._metrics,
            model_state=self._encoder_state(self.model),
            extras={"class_stats": class_stats},
        )

    def update(self) -> None:
        # receive the global feature extractor and the global common representation;
        # the personal classifier head is kept as-is
        self._global_stats = {
            class_id: (mu.to(self.device), var.to(self.device))
            for class_id, (mu, var) in self._received_messages["global_stats"].items()
        }
        self._beta = self._received_messages.get("beta", self._beta)
        self._local_stats = {}
        self.accept_parameters()
        self.solve_inner()  # alias of self.train()

    def train(self) -> None:
        """Local (stochastic) gradient descent on the objective
        :math:`L_i = f_i(w_i) + \\beta \\cdot \\mathrm{KL}[p(z^c | x) \\Vert p_i(z^c | x_i)]`
        (Eq. (7) of the paper, with per-batch class-wise feature statistics),
        where the local class-wise feature distribution is aligned with the
        global common representation via the KL divergence.
        """
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
                    if self._beta > 0 and self._features_buffer is not None:
                        # per-batch class-wise feature statistics and the KL alignment with the global stats
                        feats = self._features_buffer
                        for class_id in y.unique():
                            class_id = int(class_id.item())
                            f_c = feats[y == class_id]
                            mu_b = f_c.mean(0)
                            var_b = f_c.var(0, unbiased=False).clamp_min(1e-6)
                            # accumulate the round-level statistics
                            if class_id not in self._local_stats:
                                self._local_stats[class_id] = [
                                    torch.zeros_like(mu_b),
                                    torch.zeros_like(mu_b),
                                    torch.zeros((), device=mu_b.device),
                                ]
                            acc = self._local_stats[class_id]
                            acc[0] += mu_b * len(f_c)
                            acc[1] += (var_b + mu_b**2) * len(f_c)
                            acc[2] += len(f_c)
                            # KL[N(mu_b, var_b) || N(mu_g, var_g)], diagonal Gaussians
                            if class_id in self._global_stats:
                                mu_g, var_g = self._global_stats[class_id]
                                kl = 0.5 * (torch.log(var_g / var_b) + (var_b + (mu_b - mu_g) ** 2) / var_g - 1).sum()
                                loss = loss + self._beta * kl
                    loss.backward()
                    self.optimizer.step()
        self.lr_scheduler.step()
