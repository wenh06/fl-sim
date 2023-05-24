# A Simple Simulation Framework for Federated Learning Based on PyTorch

![formatting](https://github.com/wenh06/fl-sim/actions/workflows/check-formatting.yml/badge.svg)
![Docker CI](https://github.com/wenh06/fl-sim/actions/workflows/docker-image.yml/badge.svg?branch=docker-ci)

This repository is migrated from [fl_seminar](https://github.com/wenh06/fl_seminar/tree/master/code)

The main part of this code repository is a standalone simulation framework for federated training.

<!-- toc -->

- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Main Modules](#main-modules)
  - [Nodes](#nodes)
  - [Data Processing](#data-processing)
  - [Optimizers](#optimizers)
  - [Regularizers](#regularizers)
  - [Compression](#compression)
  - [Models](#models)
  - [Utils](#utils)
  - [Algorithms Implemented](#algorithms-implemented)

<!-- tocstop -->

## Installation

Run the following command to install the package:

```bash
pip install git+https://github.com/wenh06/fl-sim.git
```

or clone the repository and run

```bash
pip install -e .
```

## Usage Examples

<details>
<summary>Click to expand!</summary>

The following code snippet shows how to use the framework to train a model on the `FedProxFEMNIST` dataset using the `FedProx` algorithm.

```python
from fl_sim.data_processing.fedprox_femnist import FedProxFEMNIST
from fl_sim.algorithms.fedprox import (
    FedProxServer,
    FedProxClientConfig,
    FedProxServerConfig,
)

# create a FedProxFEMNIST dataset
ds = FedProxFEMNIST()
# choose a model
model = ds.candidate_models["cnn_femmist_tiny"]
# set up the server and client configurations
server_config = FedProxServerConfig(200, ds.DEFAULT_TRAIN_CLIENTS_NUM, 0.7)
client_config = FedProxClientConfig(ds.DEFAULT_BATCH_SIZE, 30)
# create a FedProxServer object
s = FedProxServer(model, ds, server_config, client_config)
# normal centralized training
s.train_centralized()
# federated training
s.train_federated()
```

</details>

## Main Modules

### [Nodes](fl_sim/nodes.py)

<details>
<summary>Click to expand!</summary>

`Node`s are the core of the simulation framework. `Node` has two subclasses: `Server` and `Client`.
The `Server` class is the base class for all servers, which acts as the coordinator of the training process, as well as maintainer of status variables.
The `Client` class is the base class for all clients.

The abstract base class `Node` provides the following basic functionalities:

- `get_detached_model_parameters`: get the model parameters of the node in a detached form.
- `aggregate_results_from_csv_log`: aggregate the experiment results from the csv log file.
- `aggregate_results_from_json_log`: aggregate the experiment results from the json log file.
- `_post_init`: post-initialization procedure, called in the end of `__init__` method.

and abstract methods or properties that need to be implemented by subclasses:

- `communicate`: communicate procedure with other (type of) nodes in each iteration.
- `update`: updating procedure in each iteration.
- `required_config_fields` (property): required fields in the configuration class, which is used to check the validity of the configuration in the `_post_init` method.

The `Server` class has signature

```python
Server(
    model: torch.nn.modules.module.Module,
    dataset: fl_sim.data_processing.fed_dataset.FedDataset,
    config: fl_sim.nodes.ServerConfig,
    client_config: fl_sim.nodes.ClientConfig,
    lazy: bool = False,
) -> None
```

providing the following additional functionalities:

- `_setup_clients`: setup (initialize) the clients, and allocate devices to them.
- `_sample_clients`: sample a subset of clients from the client pool.
- `_communicate`: execute the `communicate` method of the clients, and increment the global communication counter (`_num_communications`).
- `_update`: checks the validity messages (`_received_messages`) received from the clients, execute the `update` method of the server, and finally clears the received messages.
- `train_centralized`: centralized training procedure, mainly used for comparison.
- `train_federated`: federated training procedure, which calls the `_communicate` (to clients), wait for the clients to execute `_update` and `_communicate`, and finally calls `_update` to update the server.
- `add_parameters`: addition of parameters (values) to the server model parameters.
- `avg_parameters`: averaging the model parameters in the received messages.
- `update_gradients`: update the gradients of the server model parameters using the received gradients.
- `get_client_data`: helper function to get the data of the clients.
- `get_client_model`: helper function to get the model of the clients.
- `get_cached_metrics`: helper function to get the cached aggregated metrics of the clients stored on the server.

and **abstract properties that need to be implemented by subclasses**:

- `client_cls`: the client class used when initializing the clients via `_setup_clients`.
- `config_cls`: a dictionary of configuration classes for the server and clients, used in `__init__` method.
- `doi`: the DOI of the paper that proposes the algorithm.

The `Client` class has signature

```python
Client(
    client_id: int,
    device: torch.device,
    model: torch.nn.modules.module.Module,
    dataset: fl_sim.data_processing.fed_dataset.FedDataset,
    config: fl_sim.nodes.ClientConfig,
) -> None
```

providing the following additional functionalities:

- `_communicate`: execute the `communicate` method of the server, increment the global communication counter (`_num_communications`), and clears the cached local evaluation results.
- `_update`: execute the `update` method of the client, and clears the received messages from the server.
- `evaluate`: evaluate the model on the local test data.
- `set_parameters`: set the model parameters of the client.
- `get_gradients`: get the gradients, or norm of the gradients, of the model parameters of the client.
- `get_all_data`: helper function to get all the data of the client.

and **abstract methods that need to be implemented by subclasses**:

- `train`: training procedure of the client.

The configuration classes `ServerConfig` and `ClientConfig` are used to store the configuration of the server and clients, respectively.
These two classes are similar to a `dataclass`, but accept arbitrary additional fields. The signature of `ServerConfig` is

```python
ServerConfig(
    algorithm: str,
    num_iters: int,
    num_clients: int,
    clients_sample_ratio: float,
    txt_logger: bool = True,
    csv_logger: bool = False,
    json_logger: bool = True,
    eval_every: int = 1,
    verbose: int = 1,
    **kwargs: Any,
) -> None
```

and the signature of `ClientConfig` is

```python
ClientConfig(
    algorithm: str,
    optimizer: str,
    batch_size: int,
    num_epochs: int,
    lr: float,
    verbose: int = 1,
    **kwargs: Any,
) -> None
```

To implement a new algorithm, one needs to implement a subclass of `Server`, `Client`, `ServerConfig`, and `ClientConfig`. For example, the following implementation of FedProx is provided in the file [fedprox](fl_sim/algorithms/fedprox/_fedprox.py):

<details>
<summary>Click to expand!</summary>

```python
import warnings
from copy import deepcopy
from typing import List, Dict, Any

import torch
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from fl_sim.nodes import Server, Client, ServerConfig, ClientConfig, ClientMessage


class FedProxServerConfig(ServerConfig):
    """Server config for the FedProx algorithm.

    Parameters
    ----------
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float
        The ratio of clients to sample for each iteration.
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

    __name__ = "FedProxServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        vr: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "FedProx",
            num_iters,
            num_clients,
            clients_sample_ratio,
            vr=vr,
            **kwargs,
        )


class FedProxClientConfig(ClientConfig):
    """Client config for the FedProx algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 1e-2
        The learning rate.
    mu : float, default 0.01
        Coefficient for the proximal term.
    vr : bool, default False
        Whether to use variance reduction.
    **kwargs : dict, optional
        Additional keyword arguments:

        - ``verbose`` : int, default 1
            The verbosity level.

    """

    __name__ = "FedProxClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        mu: float = 0.01,
        vr: bool = False,
        **kwargs: Any,
    ) -> None:
        optimizer = "FedProx" if not vr else "FedProx_VR"
        if kwargs.pop("algorithm", None) is not None:
            warnings.warn(
                "The `algorithm` argument fixed to `FedProx`.", RuntimeWarning
            )
        if kwargs.pop("optimizer", None) is not None:
            warnings.warn(
                "The `optimizer` argument fixed to `FedProx` or `FedProx_VR`.",
                RuntimeWarning,
            )
        super().__init__(
            "FedProx",
            optimizer,
            batch_size,
            num_epochs,
            lr,
            mu=mu,
            vr=vr,
            **kwargs,
        )


@add_docstring(
    Server.__doc__.replace(
        "The class to simulate the server node.",
        "Server node for the FedProx algorithm.",
    )
    .replace("ServerConfig", "FedProxServerConfig")
    .replace("ClientConfig", "FedProxClientConfig")
)
class FedProxServer(Server):
    """Server node for the FedProx algorithm."""

    __name__ = "FedProxServer"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and check compatibility of server and client configs
        """
        super()._post_init()
        assert self.config.vr == self._client_config.vr

    @property
    def client_cls(self) -> type:
        return FedProxClient

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "FedProxClient") -> None:
        target._received_messages = {"parameters": self.get_detached_model_parameters()}
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

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": FedProxServerConfig,
            "client": FedProxClientConfig,
        }

    @property
    def doi(self) -> List[str]:
        return ["10.48550/ARXIV.1812.06127"]


@add_docstring(
    Client.__doc__.replace(
        "The class to simulate the client node.",
        "Client node for the FedProx algorithm.",
    ).replace("ClientConfig", "FedProxClientConfig")
)
class FedProxClient(Client):
    """Client node for the FedProx algorithm."""

    __name__ = "FedProxClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes for maintaining itermidiate states
        """
        super()._post_init()
        if self.config.vr:
            self._gradient_buffer = [
                torch.zeros_like(p) for p in self.model.parameters()
            ]
        else:
            self._gradient_buffer = None

    @property
    def required_config_fields(self) -> List[str]:
        return ["mu"]

    def communicate(self, target: "FedProxServer") -> None:
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
            self._cached_parameters = deepcopy(self._received_messages["parameters"])
        except KeyError:
            warnings.warn("No parameters received from server")
            warnings.warn("Using current model parameters as initial parameters")
            self._cached_parameters = self.get_detached_model_parameters()
        except Exception as err:
            raise err
        self._cached_parameters = [p.to(self.device) for p in self._cached_parameters]
        if (
            self.config.vr
            and self._received_messages.get("gradients", None) is not None
        ):
            self._gradient_buffer = [
                gd.clone().to(self.device)
                for gd in self._received_messages["gradients"]
            ]
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
                    self.optimizer.step(
                        local_weights=self._cached_parameters,
                        variance_buffer=self._gradient_buffer,
                    )

```

</details>

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Data Processing](fl_sim/data_processing)

<details>
<summary>Click to expand!</summary>

The module (folder) [data_processing](fl_sim/data_processing) contains code for data preprocessing, io, etc. The following datasets are included in this module:

1. [`FedCIFAR`](fl_sim/data_processing/fed_cifar.py)
2. [`FedCIFAR100`](fl_sim/data_processing/fed_cifar.py)
3. [`FedEMNIST`](fl_sim/data_processing/fed_emnist.py)
4. [`FedMNIST`](fl_sim/data_processing/fed_mnist.)
5. [`FedShakespeare`](fl_sim/data_processing/fed_shakespeare.py)
6. [`FedSynthetic`](fl_sim/data_processing/fed_synthetic.py)
7. [`FedProxFEMNIST`](fl_sim/data_processing/fedprox_femnist.py)
8. [`FedProxMNIST`](fl_sim/data_processing/fedprox_mnist.py)
9. [`FedProxSent140`](fl_sim/data_processing/fedprox_sent140.py)

Each dataset is wrapped in a class, providing the following functionalities:

1. Automatic data downloading and preprocessing
2. Data partitioning (into clients) via methods `get_dataloader`
3. A list of candidate [models] (#models) via the property `candidate_models`
4. Criterion and method for evaluating the performance of a model using its output on the dataset via the method `evaluate`
5. Several helper methods for data visualization and citation (biblatex format)

Additionally, one can get the list of `LIBSVM` datasets via

```python
pd.read_html("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/")[0]
```

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Models](fl_sim/models)

<details>
<summary>Click to expand!</summary>

The module (folder) [models](fl_sim/models) contains pre-defined (neural network) models, most of which are very simple:

1. `MLP`
2. `FedPDMLP`
3. `CNNMnist`
4. `CNNFEMnist`
5. `CNNFEMnist_Tiny`
6. `CNNCifar`
7. `RNN_OriginalFedAvg`
8. `RNN_StackOverFlow`
9. `RNN_Sent140`
10. `ResNet18`
11. `ResNet10`
12. `LogisticRegression`
13. `SVC`
14. `SVR`

Most models are proposed or suggested by previous literature.

One can call the `module_size` or `module_size_` properties to check the size (in terms of number of parameters and memory consumption respectively) of the model.

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Optimizers](fl_sim/optimizers)

<details>
<summary>Click to expand!</summary>

The module (folder) [optimizers](fl_sim/optimizers) contains optimizers for solving inner (local) optimization problems. Despite optimizers from `torch` and `torch_optimizers`, this module implements

1. `ProxSGD`
2. `FedPD_SGD`
3. `FedPD_VR`
4. `PSGD`
5. `PSVRG`
6. `pFedMe`
7. `FedProx`
8. `FedDR`

Most of the optimizers are derived from `ProxSGD`.

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Regularizers](fl_sim/regularizers)

<details>
<summary>Click to expand!</summary>

The module (folder) [regularizers](fl_sim/regularizers) contains code for regularizers for model parameters (weights).

1. `L1Norm`
2. `L2Norm`
3. `L2NormSquared`
4. `NullRegularizer`

These regularizers are subclasses of a base class `Regularizer`, and can be obtained by passing the name of the regularizer to the function `get_regularizer`. The regularizers share common methods `eval` and `prox_eval`.

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Compression](fl_sim/compressors)

<details>
<summary>Click to expand!</summary>

The module (folder) [compressors](fl_sim/compressors) contains code for constructing compressors.

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Utils](fl_sim/utils)

<details>
<summary>Click to expand!</summary>

The module (folder) [utils](fl_sim/utils) contains utility functions for [data downloading](fl_sim/utils/_download_data.py),
[training metrics logging](fl_sim/utils/loggers.py), etc.

- `TxTLogger`: A logger for logging training metrics to a text file, as well as printing them to the console, in a human-readable format.
- `CSVLogger`: A logger for logging training metrics to a CSV file. **NOT** recommended since not memory-efficient.
- `JsonLogger`: A logger for logging training metrics to a JSON file. Also can be saved as a YAML file.

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Algorithms Implemented](fl_sim/algorithms)

<details>
<summary>Click to expand!</summary>

1. [FedProx](https://github.com/litian96/FedProx) ![test-fedprox](https://github.com/wenh06/fl-sim/actions/workflows/test-fedprox.yml/badge.svg)
2. [FedOpt](https://arxiv.org/abs/2003.00295) ![test-fedopt](https://github.com/wenh06/fl-sim/actions/workflows/test-fedopt.yml/badge.svg)
3. [pFedMe](https://github.com/CharlieDinh/pFedMe) ![test-pfedme](https://github.com/wenh06/fl-sim/actions/workflows/test-pfedme.yml/badge.svg)
4. [FedSplit](https://arxiv.org/abs/2005.05238) ![test-fedsplit](https://github.com/wenh06/fl-sim/actions/workflows/test-fedsplit.yml/badge.svg)
5. [FedDR](https://github.com/unc-optimization/FedDR) ![test-feddr](https://github.com/wenh06/fl-sim/actions/workflows/test-feddr.yml/badge.svg)
6. [FedPD](https://github.com/564612540/FedPD/) ![test-fedpd](https://github.com/wenh06/fl-sim/actions/workflows/test-fedpd.yml/badge.svg)
7. [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html) ![test-scaffold](https://github.com/wenh06/fl-sim/actions/workflows/test-scaffold.yml/badge.svg)
8. [ProxSkip](https://proceedings.mlr.press/v162/mishchenko22b.html) ![test-proxskip](https://github.com/wenh06/fl-sim/actions/workflows/test-proxskip.yml/badge.svg)
9. [Ditto](https://arxiv.org/abs/2012.04221) ![test-ditto](https://github.com/wenh06/fl-sim/actions/workflows/test-ditto.yml/badge.svg)

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>
