# A Simple Simulation Framework for Federated Learning Based on PyTorch

[![formatting](https://github.com/wenh06/fl-sim/actions/workflows/check-formatting.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/check-formatting.yml)
[![Docker CI](https://github.com/wenh06/fl-sim/actions/workflows/docker-image.yml/badge.svg?branch=docker-ci)](https://github.com/wenh06/fl-sim/actions/workflows/docker-image.yml)
[![PyTest](https://github.com/wenh06/fl-sim/actions/workflows/run-pytest.yml/badge.svg?branch=dev)](https://github.com/wenh06/fl-sim/actions/workflows/run-pytest.yml)
[![codecov](https://codecov.io/gh/wenh06/fl-sim/branch/master/graph/badge.svg?token=B36FC6VIFD)](https://codecov.io/gh/wenh06/fl-sim)

[English Version](README.md)

项目链接:

- 源代码: [GitHub](https://github.com/wenh06/fl-sim) | [gitee](https://gitee.com/wenh06/fl-sim)
- 文档（正在完善）: [GitHub Pages](https://wenh06.github.io/fl-sim/)  [![gh-page status](https://github.com/wenh06/fl-sim/actions/workflows/docs-test-publish.yml/badge.svg)](https://github.com/wenh06/fl-sim/actions/workflows/docs-test-publish.yml) | [Read the Docs](http://fl-sim.rtfd.io/)  [![RTD Status](https://readthedocs.org/projects/fl-sim/badge/?version=latest)](https://fl-sim.readthedocs.io/en/latest/?badge=latest)

本仓库迁移自 [fl_seminar](https://github.com/wenh06/fl_seminar/tree/master/code)，
主体部分是一个基于 PyTorch 的简单的联邦学习仿真框架。

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

[^1]: FedAvg 是作为 FedOpt 的特例进行实现的。
[^2]: 包括 FedAdam, FedYogi, FedAdagrad 等算法。

标准测试效果图：

[Client sample ratio 10%](https://deep-psp.tech/FLSim/standard-test-ratio-10-val-acc.svg)
[Client sample ratio 30%](https://deep-psp.tech/FLSim/standard-test-ratio-30-val-acc.svg)
[Client sample ratio 70%](https://deep-psp.tech/FLSim/standard-test-ratio-70-val-acc.svg)
[Client sample ratio 100%](https://deep-psp.tech/FLSim/standard-test-ratio-100-val-acc.svg)

- :heavy_check_mark: 算法在标准测试用例上的效果符合预期。
- :heavy_check_mark: :question: 算法在标准测试用例上的效果 **低于** 预期。
- :question: 算法暂未在标准测试用例上进行测试。
- :interrobang: 算法在标准测试用例上的 **发散** ，相关的算法实现需要进一步检查。

## 主要模块

### [Nodes](fl_sim/nodes.py)

<details>
<summary>点击展开</summary>

`Node` 类是本仿真框架的核心。`Node` 有两个子类： `Server` 和 `Client`。
`Server` 类是所有联邦学习算法中心节点的基类，它在训练过程中充当协调者和状态变量维护者的角色。
`Client` 类是所有联邦学习算法子节点的基类。

抽象基类 `Node` 提供了以下基本功能：

- `get_detached_model_parameters`: 获取节点上模型参数的副本。
- `compute_gradients`: 计算指定模型参数（默认为节点上的当前模型参数）在指定数据（默认为节点上的训练数据）上的梯度。
- `get_gradients`: 获取当前节点上模型当前的梯度，或者梯度的范数。
- `get_norm`: 计算一个 tensor 或者 array 的范数。
- `set_parameters`: 设置节点上模型参数。
- ~~`aggregate_results_from_csv_log`: 从 csv 日志文件中聚合实验结果。~~
- `aggregate_results_from_json_log`: 从 json 日志文件中聚合实验结果。

以及需要子类实现的抽象方法或属性：

- `communicate`: 在每一轮训练中，与另一种类型节点进行通信的方法 （子节点 -> 中心节点 或者 中心节点 -> 子节点）。
- `update`: 在每一轮训练中，更新节点状态的方法。
- `required_config_fields` (property): 需要在配置类中指定的必要字段，用于在 `_post_init` 方法中检查配置的有效性。
- `_post_init`: 在 `__init__` 方法的最后调用的后初始化方法，用于在 `__init__` 方法中检查配置的有效性。

`Server` 类的签名（signature）为

```python
Server(
    model: torch.nn.modules.module.Module,
    dataset: fl_sim.data_processing.fed_dataset.FedDataset,
    config: fl_sim.nodes.ServerConfig,
    client_config: fl_sim.nodes.ClientConfig,
    lazy: bool = False,
) -> None
```

`Server` 类提供以下额外的方法（method）或属性（property）：

- `_setup_clients`: 初始化客户端，为客户端分配计算资源。
- `_sample_clients`: 从所有子节点中随机抽取一定数量的子节点。
- `_communicate`: 执行子节点的 `communicate` 方法，并更新全局通信计数器（`_num_communications`）。
- `_update`: 检查从子节点接收到消息（`_received_messages`）的有效性，执行中心节点的 `update` 方法，最后清除所有从子节点接收到的消息。
- `train`: 联邦训练主循环，根据传入的 `mode` 参数调用 `train_centralized` 或者 `train_federated` 或者 `train_local` 方法。
- `train_centralized`: 中心化训练，主要用于对比。
- `train_federated`: 联邦训练，调用 `_communicate` 方法（与子节点通信），等待子节点执行 `_update` 和 `_communicate` 方法，最后调用 `_update` 方法（更新中心节点）。
- `train_local`: 本地训练，调用子节点的 `train` 方法，**不** 与中心节点通信。
- `add_parameters`: 中心节点模型参数的增量更新。
- `avg_parameters`: 将从子节点接收到的模型参数进行平均。
- `update_gradients`: 使用从子节点接收到的梯度更新中心节点模型的梯度。
- `get_client_data`: 获取特定子节点的数据。
- `get_client_model`: 获取特定子节点的模型。
- `get_cached_metrics`: 获取中心节点缓存的每一次训练循环的模型评估指标。
- `_reset`: 将中心节点重置为初始状态。在执行新的训练过程之前，将检查 `_complete_experiment` 标志。如果为 `True`，将调用此方法重置中心节点。
- `is_convergent` (property): 检查训练过程是否收敛。目前，此属性 **未** 完全实现。

以及 **需要子类实现的抽象属性**：

- `client_cls`: the client class used when initializing the clients via `_setup_clients`.
- `config_cls`: a dictionary of configuration classes for the server and clients, used in `__init__` method.
- `doi`: the DOI of the paper that proposes the algorithm.

`Client` 类的签名为

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
`Client` 类还提供以下额外的方法：

- `_communicate`: 执行子节点的 `communicate` 方法，并更新子节点上的通信计数器（`_num_communications`），并清除缓存的（上一循环）子节点上模型评测结果。
- `_update`: 执行子节点的 `update` 方法，并清除从中心节点接收到的消息。
- `evaluate`: 利用子节点上的测试数据，评测子节点上的模型。
- `get_all_data`: 获取子节点上的所有数据。

以及 **需要子类实现的抽象方法**：

- `train`: 子节点的训练循环。

配置类（config class）是用于存储服务器和客户端配置的类。这两个类类似于 [`dataclass`](https://docs.python.org/3/library/dataclasses.html)，但是可以接受任意额外的字段。`ServerConfig` 的签名为

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

`ClientConfig` 的签名为

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

实现 **新的联邦学习算法** 的方法：需要实现 `Server` 和 `Client` 的子类，以及 `ServerConfig` 和 `ClientConfig` 的子类。如下的例子，是取自 [`FedProx`](fl_sim/algorithms/fedprox/_fedprox.py) 算法的实现：

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

:point_right: [返回目录](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Data Processing](fl_sim/data_processing)

<details>
<summary>点击展开</summary>

[data_processing](fl_sim/data_processing) 模块包含数据预处理、IO 等代码，其中包含以下数据集：

1. [`FedCIFAR`](fl_sim/data_processing/fed_cifar.py)
2. [`FedCIFAR100`](fl_sim/data_processing/fed_cifar.py)
3. [`FedEMNIST`](fl_sim/data_processing/fed_emnist.py)
4. [`FedMNIST`](fl_sim/data_processing/fed_mnist.)
5. [`FedShakespeare`](fl_sim/data_processing/fed_shakespeare.py)
6. [`FedSynthetic`](fl_sim/data_processing/fed_synthetic.py)
7. [`FedProxFEMNIST`](fl_sim/data_processing/fedprox_femnist.py)
8. [`FedProxMNIST`](fl_sim/data_processing/fedprox_mnist.py)
9. [`FedProxSent140`](fl_sim/data_processing/fedprox_sent140.py)

以上每一个数据集都被封装在一个类中，提供以下功能：

1. 数据集的自动下载和预处理
2. 数据集的切分（分配给子节点）方法 `get_dataloader`
3. 预置了一系列候选 [模型](#models)，可以通过 `candidate_models` 属性获取
4. 基于模型预测值的 `evaluate` 方法，可以评测模型在数据集上的性能
5. 一些辅助方法，用于数据可视化和参考文献的获取（biblatex 格式）

此外， `LIBSVM` 数据集列表可以通过如下语句获取

```python
pd.read_html("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/")[0]
```

**更新**: 一部分计算机视觉数据集的训练集支持动态数据增强。基类 `FedVisionDataset` 的签名为

```python
FedVisionDataset(
    datadir: Union[str, pathlib.Path, NoneType] = None,
    transform: Union[str, Callable, NoneType] = "none",
) -> None
```

通过将 `transform` 参数设置为 `"none"` （这也是 `transform` 参数的默认值），训练集将被封装在一个静态的 `TensorDataset` 中。通过将 `transform` 参数设置为 `None`，训练集将使用内置的动态数据增强，例如 `FedCIFAR100` 使用 `torchvision.transforms.RandAugment`。

**注意**，大部分计算机视觉的联邦数据集包含的数据都是经过预处理后的而不是原始像素值，因此不支持使用 `torchvision.transforms` 进行动态数据增强。

:point_right: [返回目录](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Models](fl_sim/models)

<details>
<summary>点击展开</summary>

[models](fl_sim/models) 模块包含预定义的（神经网络）模型，其中大部分结构都非常简单：

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

以上大部分模型都是之前文献中使用过的，或是基于此进行修改的。

通过调用 `model_size` 或 `model_size_` 属性可以获取模型的大小（参数数量和内存占用）。

:point_right: [返回目录](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Optimizers](fl_sim/optimizers)

<details>
<summary>点击展开</summary>

[optimizers](fl_sim/optimizers) 模块包含用于解决联邦优化问题内循环（子节点上的）优化问题的优化器。除了 `torch` 和 `torch_optimizers` 中的优化器外，本模块实现了以下优化器：

1. `ProxSGD`
2. `FedPD_SGD`
3. `FedPD_VR`
4. `PSGD`
5. `PSVRG`
6. `pFedMe`
7. `FedProx`
8. `FedDR`

其中大部分都是基于 `ProxSGD` 的变体，即目标是带有临近项的优化问题。

:point_right: [返回目录](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Regularizers](fl_sim/regularizers)

<details>
<summary>点击展开</summary>

[regularizers](fl_sim/regularizers) 模块包含用于对模型参数进行正则化的正则化项（用类来实现）。正则化项的目的是防止模型过拟合，从而提高模型的泛化能力。本模块实现了以下正则化项：

1. `L1Norm`
2. `L2Norm`
3. `L2NormSquared`
4. `NullRegularizer`

以上的正则化项都是基类 `Regularizer` 的子类，可以通过将正则化项的名称传递给函数 `get_regularizer` 来获取。正则化项都有 `eval` 和 `prox_eval` 两个方法，分别用于计算正则化项的值和其临近项的值。

:point_right: [返回目录](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Compression](fl_sim/compressors)

<details>
<summary>点击展开</summary>

[compressors](fl_sim/compressors) 模块包含了模型参数压缩器的实现。压缩器的目的是减少模型参数的传输量，从而减少通信开销。

:point_right: [返回目录](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Utils](fl_sim/utils)

<details>
<summary>点击展开</summary>

[utils](fl_sim/utils) 模块包含了一些工具函数，例如 [数据下载](fl_sim/utils/_download_data.py)、 [日志记录](fl_sim/utils/loggers.py)、 [可视化](fl_sim/utils/viz.py) 等。

- `TxTLogger`: 用于将训练指标记录到文本文件中，同时也会在控制台以适合人类阅读习惯的格式打印出来。
- ~~`CSVLogger`: 用于将训练指标记录到 CSV 文件中。**不推荐使用**，因为存储消耗较大。~~
- `JsonLogger`: 用于将训练指标记录到 JSON 文件中。也可以保存为 YAML 文件。

:point_right: [返回目录](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

</details>

### [Visualization Panel](fl_sim/utils/viz.py)

本框架实现了一个可视化面板，用于可视化联邦学习算法的训练结果。它基于 `ipywidgets` 和 `matplotlib` 进行开发，可以在 Jupyter notebook 中使用。它具有以下功能：

1. 自动搜索并显示指定目录中完整实验的日志文件。
2. 自动解析日志文件，并将训练指标进行聚合，利用 matplotlib 生成曲线。
3. 支持对绘制的图像进行交互式操作，包括缩放、字体选择、曲线平滑等。
4. 支持将绘制的图像保存为 PDF/SVG/PNG/JPEG/PS 等格式的文件。
5. 支持将不同实验曲线进行合并，例如可以将使用不同随机数种子的 `FedAvg` 算法的数值曲线合并成一条均值曲线。合并后的曲线可以选择是否显示标准差、标准误差、分位数、四分位距等误差范围。

下面的 GIF （使用 [ScreenToGif](https://github.com/NickeManarin/ScreenToGif) 制作生成）是可视化面板的演示示例：

<img src="https://raw.githubusercontent.com/wenh06/fl-sim/master/docs/source/_static/images/panel-demo.gif" alt="FL-SIM Panel Demo GIF" style="display: block; margin: 0 auto;" />

**注意：** 若希望在 Linux 系统下（例如 Ubuntu）上使用 Windows 字体，可以执行以下命令获取相关字体：

```bash
sudo apt install ttf-mscorefonts-installer
sudo fc-cache -fv
```

## 命令行接口

本仿真框架提供了命令行接口（CLI），用于一次性执行多个联邦学习实验。命令行接口只有一个参数，即实验的配置文件（YAML 格式）路径。配置文件的示例可以在 [example-configs](example-configs) 文件夹中找到。例如，在 [all-alg-fedprox-femnist.yml](example-configs/all-alg-fedprox-femnist.yml) 文件中，我们写入了如下的配置：

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

`strategy` 字段指定了网格搜索的策略。
`algorithm` 字段指定了联邦学习算法的超参数：
其中 `name` 字段指定了算法的名称，`server` 字段指定了中心节点的超参数，`client` 字段指定了子节点的超参数。
`dataset` 字段指定了实验使用的数据集，`model` 字段指定了实验使用的模型。

## 自定义算法的实现

利用本仿真框架实现的注册机制（registration functions），可以很方便地实现自定义的联邦学习算法，数据集，优化器等。例如，在文件 [custom_confi.yml](test-files/custom_conf.yml) 中，我们写入了如下的配置：

- `algorithm.name: test-files/custom_alg.Custom`
- `dataset.name: test-files/custom_dataset.CustomFEMNIST`

其中 [`test-files/custom_alg.py`](test-files/custom_alg.py)， [`test-files/custom_dataset.py`](test-files/custom_dataset.py) 分别是自定义算法和自定义数据集的文件，`Custom` 是自定义算法的名称，`CustomFEMNIST` 是自定义数据集的名称。我们可以在本仓库的根目录下执行以下命令来执行仿真数值试验

```bash
fl-sim test-files/custom_conf.yml
```

若 `algorithm.name` 和 `dataset.name` 是绝对路径，则可以在任意位置执行该命令。

### 自定义联邦学习算法

在文件 [test-files/custom_alg.py](test-files/custom_alg.py) 中，我们实现了一个自定义的联邦学习算法 `Custom`，该算法的实现细节如下：
将算法的超参数配置写入 `CustomServerConfig` 和 `CustomClientConfig` 类中，这两个类分别继承了 `ServerConfig` 和 `ClientConfig` 类。
将算法的实现写入 `CustomServer` 和 `CustomClient` 类中，这两个类分别继承了 `Server` 和 `Client` 类。同时，利用装饰器 `register_algorithm`，我们将 `CustomServerConfig`，`CustomClientConfig`，`CustomServer`，`CustomClient` 注册到了本仿真框架中，例如：

```python
@register_algorithm()
@add_docstring(server_config_kw_doc, "append")
class CustomServerConfig(ServerConfig):
    ...

```

之后在利用命令行接口执行仿真数值试验时，就可以通过 `algorithm.name` 指定 `Custom` 算法。

### Custom Datasets

类似地，我们可以实现自定义的联邦数据集。在文件 [test-files/custom_dataset.py](test-files/custom_dataset.py) 中，我们实现了一个自定义的联邦数据集 `CustomFEMNIST`，其继承了 `FEMNIST` 类。同时，利用装饰器 `register_dataset`，我们将 `CustomFEMNIST` 注册到了本仿真框架中。

### Custom Optimizers

自定义的优化器也可以通过类似的方式实现，即将其实现为 `torch.optim.Optimizer` 的子类，并利用装饰器 `register_optimizer` 将其注册到本仿真框架中。
