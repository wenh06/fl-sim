# A Simple Simulation Framework for Federated Learning Based on PyTorch

[![formatting](https://github.com/wenh06/fl-sim/actions/workflows/check-formatting.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/check-formatting.yml)
[![Docker CI](https://github.com/wenh06/fl-sim/actions/workflows/docker-image.yml/badge.svg?branch=docker-ci)](https://github.com/wenh06/fl-sim/actions/workflows/docker-image.yml)
[![PyTest](https://github.com/wenh06/fl-sim/actions/workflows/run-pytest.yml/badge.svg?branch=dev)](https://github.com/wenh06/fl-sim/actions/workflows/run-pytest.yml)
[![codecov](https://codecov.io/gh/wenh06/fl-sim/branch/master/graph/badge.svg?token=B36FC6VIFD)](https://codecov.io/gh/wenh06/fl-sim)

[English Version](README.md)

本仓库迁移自 [fl_seminar](https://github.com/wenh06/fl_seminar/tree/master/code)，主体部分是一个基于 PyTorch 的简单的联邦学习仿真框架。

文档地址（正在完善）：

- [GitHub Pages](https://wenh06.github.io/fl-sim/)  [![gh-page status](https://github.com/wenh06/fl-sim/actions/workflows/docs-test-publish.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/docs-test-publish.yml)
- [Read the Docs](http://fl-sim.rtfd.io/)  [![RTD Status](https://readthedocs.org/projects/fl-sim/badge/?version=latest)](https://fl-sim.readthedocs.io/en/latest/?badge=latest)

<!-- toc -->

- [安装](#安装)
- [示例](#示例)
- [复现的算法](#复现的算法)
- [主要模块](#主要模块)
  - [Nodes](#nodes)
  - [Data Processing](#data-processing)
  - [Optimizers](#optimizers)
  - [Regularizers](#regularizers)
  - [Compression](#compression)
  - [Models](#models)
  - [Utils](#utils)
  - [Visualization Panel](#visualization-panel)
- [命令行接口](#命令行接口)
- [自定义算法的实现](#自定义算法的实现)

<!-- tocstop -->

## 安装

可以在命令行中使用以下命令安装：

```bash
pip install git+https://github.com/wenh06/fl-sim.git
```

或者，可以先将仓库克隆到本地，然后在仓库根目录下使用以下命令安装：

```bash
pip install -e .
```

## 示例

<details>
<summary>点击展开</summary>

以下代码片段展示了如何使用框架在 `FedProxFEMNIST` 数据集上使用 `FedProx` 算法训练模型。

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

## [复现的算法](fl_sim/algorithms)

| 算法  | 文章 | 源仓库 | Action 状态 | [标准测试用例](example-configs/all-alg-fedprox-femnist.yml)上的效果 |
| ---------- | ----- | -------- | ------------- | --------------------------------------------------------------------- |
| [FedAvg[^1]](fl_sim/algorithms/fedopt/) | [AISTATS2017](https://proceedings.mlr.press/v54/mcmahan17a.html) | N/A | [![test-fedopt](https://github.com/wenh06/fl-sim/actions/workflows/test-fedopt.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-fedopt.yml) | :heavy_check_mark: |
| [FedOpt[^2]](fl_sim/algorithms/fedopt/) | [arXiv:2003.00295](https://arxiv.org/abs/2003.00295) | N/A | [![test-fedopt](https://github.com/wenh06/fl-sim/actions/workflows/test-fedopt.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-fedopt.yml) | :heavy_check_mark: |
| [FedProx](fl_sim/algorithms/fedprox/)   | [MLSys2020](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html) | [GitHub](https://github.com/litian96/FedProx) | [![test-fedprox](https://github.com/wenh06/fl-sim/actions/workflows/test-fedprox.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-fedprox.yml)  | :heavy_check_mark: :question: |
| [pFedMe](fl_sim/algorithms/pfedme/)     | [NeurIPS2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f4f1f13c8289ac1b1ee0ff176b56fc60-Abstract.html) | [GitHub](https://github.com/CharlieDinh/pFedMe)     | [![test-pfedme](https://github.com/wenh06/fl-sim/actions/workflows/test-pfedme.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-pfedme.yml) | :interrobang: |
| [FedSplit](fl_sim/algorithms/fedsplit/) | [NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/4ebd440d99504722d80de606ea8507da-Abstract.html) | N/A | [![test-fedsplit](https://github.com/wenh06/fl-sim/actions/workflows/test-fedsplit.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-fedsplit.yml) | :heavy_check_mark: :question: |
| [FedDR](fl_sim/algorithms/feddr/)       | [NeurIPS2021](https://papers.nips.cc/paper/2021/hash/fe7ee8fc1959cc7214fa21c4840dff0a-Abstract.html) | [GitHub](https://github.com/unc-optimization/FedDR) | [![test-feddr](https://github.com/wenh06/fl-sim/actions/workflows/test-feddr.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-feddr.yml) | :interrobang: |
| [FedPD](fl_sim/algorithms/fedpd/)       | [IEEE Trans. Signal Process](https://ieeexplore.ieee.org/document/9556559) | [GitHub](https://github.com/564612540/FedPD/) | [![test-fedpd](https://github.com/wenh06/fl-sim/actions/workflows/test-fedpd.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-fedpd.yml) | :interrobang: |
| [SCAFFOLD](fl_sim/algorithms/scaffold/) | [PMLR](https://proceedings.mlr.press/v119/karimireddy20a.html) | N/A | [![test-scaffold](https://github.com/wenh06/fl-sim/actions/workflows/test-scaffold.yml/badge.svg)]((https://github.com/wenh06/fl-sim/actions/workflows/test-scaffold.yml)) | :heavy_check_mark: :question: |
| [ProxSkip](fl_sim/algorithms/proxskip/) | [PMLR](https://proceedings.mlr.press/v162/mishchenko22b.html) | N/A | [![test-proxskip](https://github.com/wenh06/fl-sim/actions/workflows/test-proxskip.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-proxskip.yml) | :heavy_check_mark: :question: |
| [Ditto](fl_sim/algorithms/ditto/)       | [PMLR](https://proceedings.mlr.press/v139/li21h.html) | [GitHub](https://github.com/litian96/ditto) | [![test-ditto](https://github.com/wenh06/fl-sim/actions/workflows/test-ditto.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-ditto.yml) | :heavy_check_mark: |
| [IFCA](fl_sim/algorithms/ifca/)         | [NeurIPS2020](https://papers.nips.cc/paper_files/paper/2020/hash/e32cc80bf07915058ce90722ee17bb71-Abstract.html) | [GitHub](https://github.com/jichan3751/ifca) | [![test-ifca](https://github.com/wenh06/fl-sim/actions/workflows/test-ifca.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-ifca.yml) | :heavy_check_mark: |
| [pFedMac](fl_sim/algorithms/pfedmac/)   | [arXiv:2107.05330](https://arxiv.org/abs/2107.05330) | N/A | [![test-pfedmac](https://github.com/wenh06/fl-sim/actions/workflows/test-pfedmac.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-pfedmac.yml) | :interrobang: |
| [FedDyn](fl_sim/algorithms/feddyn/)   | [ICLR2021](https://openreview.net/forum?id=B7v4QMR6Z9w) | N/A | [![test-feddyn](https://github.com/wenh06/fl-sim/actions/workflows/test-feddyn.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-feddyn.yml) | :question: |
| [APFL](fl_sim/algorithms/apfl/)   | [arXiv:2003.13461](https://arxiv.org/abs/2003.13461) | N/A | [![test-apfl](https://github.com/wenh06/fl-sim/actions/workflows/test-apfl.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/test-apfl.yml) | :question: |

[^1]: FedAvg is implemented as a special case of FedOpt.
[^2]: Including FedAdam, FedYogi, FedAdagrad.

Standard Test Status Images:

[Client sample ratio 10%](https://deep-psp.tech/FLSim/standard-test-ratio-10-val-acc.svg)
[Client sample ratio 30%](https://deep-psp.tech/FLSim/standard-test-ratio-30-val-acc.svg)
[Client sample ratio 70%](https://deep-psp.tech/FLSim/standard-test-ratio-70-val-acc.svg)
[Client sample ratio 100%](https://deep-psp.tech/FLSim/standard-test-ratio-100-val-acc.svg)

- :heavy_check_mark: means that the algorithm on the standard test cases reaches expected performance.
- :heavy_check_mark: :question: means that the algorithm on the standard test cases is **below** expected performance.
- :question: means that the algorithm has not yet been tested on the standard test cases.
- :interrobang: means that the algorithm on the standard test cases **does not** converge, and the implementation has to be checked.

## 主要模块

### [Nodes](fl_sim/nodes.py)

<details>
<summary>点击展开</summary>

`Node`s are the core of the simulation framework. `Node` has two subclasses: `Server` and `Client`.
The `Server` class is the base class for all servers, which acts as the coordinator of the training process, as well as maintainer of status variables.
The `Client` class is the base class for all clients.

The abstract base class `Node` provides the following basic functionalities:

- `get_detached_model_parameters`: get the model parameters of the node in a detached form.
- `compute_gradients`: compute the gradients at specified model parameters (default: current model parameters on the node) using specified data (default: training data on the node).
- `get_gradients`: get the gradients, or norm of the gradients, of the model parameters of the node.
- `get_norm`: get the norm of given tensors or numpy arrays.
- `set_parameters`: set the model parameters of the node.
- ~~`aggregate_results_from_csv_log`: aggregate the experiment results from the csv log file.~~
- `aggregate_results_from_json_log`: aggregate the experiment results from the json log file.

and abstract methods or properties that need to be implemented by subclasses:

- `communicate`: communicate procedure with other (type of) nodes in each iteration.
- `update`: updating procedure in each iteration.
- `required_config_fields` (property): required fields in the configuration class, which is used to check the validity of the configuration in the `_post_init` method.
- `_post_init`: post-initialization procedure, called in the end of `__init__` method, used in companion with `required_config_fields` to check the validity of the configuration.

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

providing the following additional functionalities or properties:

- `_setup_clients`: setup (initialize) the clients, and allocate devices to them.
- `_sample_clients`: sample a subset of clients from the client pool.
- `_communicate`: execute the `communicate` method of the clients, and increment the global communication counter (`_num_communications`).
- `_update`: checks the validity messages (`_received_messages`) received from the clients, execute the `update` method of the server, and finally clears the received messages.
- `train`: the main training procedure, which calls either `train_centralized` or `train_federated` depending on the argument `mode` passed to this method.
- `train_centralized`: centralized training procedure, mainly used for comparison.
- `train_federated`: federated training procedure, which calls the `_communicate` (to clients), wait for the clients to execute `_update` and `_communicate`, and finally calls `_update` to update the server.
- `train_local`: local training procedure, which calls the `train` method of the clients **without** communication with the server.
- `add_parameters`: addition of parameters (values) to the server model parameters.
- `avg_parameters`: averaging the model parameters in the received messages.
- `update_gradients`: update the gradients of the server model parameters using the received gradients.
- `get_client_data`: helper function to get the data of the clients.
- `get_client_model`: helper function to get the model of the clients.
- `get_cached_metrics`: helper function to get the cached aggregated metrics of the clients stored on the server.
- `_reset`: reset the server to the initial state. Before carrying out a new training process, the flag `_complete_experiment` will be checked. If it is `True`, this method will be called to reset the server.
- `is_convergent` (property): check whether the training process has converged. Currently, this property is **NOT** fully implemented.

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
<summary>点击展开</summary>

```python
import warnings
from copy import deepcopy
from typing import List, Dict, Any

import torch
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from fl_sim.nodes import Server, Client, ServerConfig, ClientConfig, ClientMessage
from fl_sim.algorithms import register_algorithm


@register_algorithm("FedProx")
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
        - ``extra_observes`` : List[str], optional
            Extra attributes to observe during training.
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


@register_algorithm("FedProx")
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

        - ``scheduler`` : dict, optional
            The scheduler config.
            None for no scheduler, using constant learning rate.
        - ``extra_observes`` : List[str], optional
            Extra attributes to observe during training,
            which would be recorded in evaluated metrics,
            sent to the server, and written to the log file.
        - ``verbose`` : int, default 1
            The verbosity level.
        - ``latency`` : float, default 0.0
            The latency of the client.
            Not used in the current version, reserved for future use.

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


@register_algorithm("FedProx")
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


@register_algorithm("FedProx")
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
<summary>点击展开</summary>

The module (folder) [data_processing](fl_sim/data_processing) contains code for data preprocessing, io, etc.
The following datasets are included in this module:

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

**NEW**: Part of the vision datasets support dynamic data augmentation for the train subset. The base class `FedVisionDataset` has signature

```python
FedVisionDataset(
    datadir: Union[str, pathlib.Path, NoneType] = None,
    transform: Union[str, Callable, NoneType] = "none",
) -> None
```

By setting `transform="none"` (default), the train subset is wrapped with a static `TensorDataset`. By setting `transform=None`, the train subset uses built-in dynamic augmentation, for example `FedCIFAR100` uses `torchvision.transforms.RandAugment`.

**NOTE** that most of the federated vision datasets are provided with processed values rather than raw pixels, hence not supporting dynamic data augmentation using `torchvision.transforms`.

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Models](fl_sim/models)

<details>
<summary>点击展开</summary>

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
<summary>点击展开</summary>

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
<summary>点击展开</summary>

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
<summary>点击展开</summary>

The module (folder) [compressors](fl_sim/compressors) contains code for constructing compressors.

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Utils](fl_sim/utils)

<details>
<summary>点击展开</summary>

The module (folder) [utils](fl_sim/utils) contains utility functions for [data downloading](fl_sim/utils/_download_data.py),
[training metrics logging](fl_sim/utils/loggers.py), [experiment visualization](fl_sim/utils/viz.py), etc.

- `TxTLogger`: A logger for logging training metrics to a text file, as well as printing them to the console, in a human-readable format.
- `CSVLogger`: A logger for logging training metrics to a CSV file. **NOT** recommended since not memory-efficient.
- `JsonLogger`: A logger for logging training metrics to a JSON file. Also can be saved as a YAML file.

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Visualization Panel](fl_sim/utils/viz.py)

The visualization panel is a GUI for visualizing the training results of federated learning algorithms.
It is based on `ipywidgets` and `matplotlib`, and can be used in Jupyter notebooks. It has the following features:

1. Automatically search and display the log files of complete experiments in the specified directory.
2. Automatically decode the log files and aggregate the training metrics into curves in a matplotlib figure.
3. Support interactive operations on the figure, including zooming, font family selection, curve smoothing, etc.
4. Support saving the figure as a PDF/SVG/PNG/JPEG/PS file.
5. Support curves merging via tags (e.g. `FedAvg` and `FedProx` can be merged into a single curve `FedAvg/FedProx`) into mean curves with error bounds (standard deviation, standard error of the mean, quantiles, interquartile range, etc.).

The following GIF (created using [ScreenToGif](https://github.com/NickeManarin/ScreenToGif)) shows a demo of the visualization panel:

<img src="https://raw.githubusercontent.com/wenh06/fl-sim/master/images/panel-demo.gif" alt="FL-SIM Panel Demo GIF" style="display: block; margin: 0 auto;" />

**NOTE:** to use Windows fonts on a Linux machine (e.g. Ubuntu), one can execute the following commands:

```bash
sudo apt install ttf-mscorefonts-installer
sudo fc-cache -fv
```

## 命令行接口

A command line interface (CLI) is provided for running multiple federated learning experiments.
The only argument is the path to the configuration file (in YAML format) for the experiments.
Examples of configuration files can be found in the [example-configs](example-configs) folder.
For example, in the [all-alg-fedprox-femnist.yml](example-configs/all-alg-fedprox-femnist.yml) file, we have

<details>
<summary>点击展开</summary>

```yaml
# Example config file for fl-sim command line interface

strategy:
  matrix:
    algorithm:
    - Ditto
    - FedDR
    - FedAvg
    - FedAdam
    - FedProx
    - FedPD
    - FedSplit
    - IFCA
    - pFedMac
    - pFedMe
    - ProxSkip
    - SCAFFOLD
    clients_sample_ratio:
    - 0.1
    - 0.3
    - 0.7
    - 1.0

algorithm:
  name: ${{ matrix.algorithm }}
  server:
    num_clients: null
    clients_sample_ratio: ${{ matrix.clients_sample_ratio }}
    num_iters: 100
    p: 0.3  # for FedPD, ProxSkip
    lr: 0.03  # for SCAFFOLD
    num_clusters: 10  # for IFCA
    log_dir: all-alg-fedprox-femnist
  client:
    lr: 0.03
    num_epochs: 10
    batch_size: null  # null for default batch size
    scheduler:
      name: step  # StepLR
      step_size: 1
      gamma: 0.99
dataset:
  name: FedProxFEMNIST
  datadir: null  # default dir
  transform: none  # none for static transform (only normalization, no augmentation)
model:
  name: cnn_femmist_tiny
seed: 0
```

</details>

The `strategy` section specifies the grid search strategy;
the `algorithm` section specifies the hyperparameters of the federated learning algorithm:
`name` is the name of the algorithm, `server` specifies the hyperparameters of the server,
and `client` specifies the hyperparameters of the client;
the `dataset` section specifies the dataset, and the `model` section specifies the named model (ref. the `candidate_models` property of the dataset classes) to be used.

## 自定义算法的实现

One can implement custom federated learning algorithms, datasets, optimizers with corresponding registration functions.

For example, in the [custom_confi.yml](test-files/custom_conf.yml) file, we set

- `algorithm.name: test-files/custom_alg.Custom`
- `dataset.name: test-files/custom_dataset.CustomFEMNIST`

where [`test-files/custom_alg.py`](test-files/custom_alg.py) and [`test-files/custom_dataset.py`](test-files/custom_dataset.py) are the files containing the custom algorithm and dataset, respectively, and `Custom` is the name of the custom algorithm and `CustomFEMNIST` is the name of the custom dataset. One can run the following command to start the simulation:

```bash
fl-sim test-files/custom_conf.yml
```

in the root directory of this repository. If `algorithm.name` and `dataset.name` were changed to absolute paths, then one can run the command from any place.

### Custom Federated Learning Algorithms

In the [test-files/custom_alg.py](test-files/custom_alg.py) file, we implement a custom federated learning algorithm `Custom` via subclassing the 4 classes `ServerConfig`, `ClientConfig`, `Server`, and `Client`, and use the `register_algorithm` decorator to register the algorithm. For example, the `ServerConfig` class is defined as follows:

```python
@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class CustomServerConfig(ServerConfig):
    ...

```

### Custom Datasets

In the [test-files/custom_dataset.py](test-files/custom_dataset.py) file, we implement a custom dataset `CustomFEMNIST` via subclassing the `FEMNIST` class and use the `register_dataset` decorator to register the dataset.

### Custom Optimizers

One can implement custom optimizers via subclassing the `torch.optim.Optimizer` class and use the `register_optimizer` decorator to register the optimizer.
