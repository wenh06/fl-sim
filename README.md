# A Simple Simulation Framework for Federated Learning Based on PyTorch

![formatting](https://github.com/wenh06/fl-sim/actions/workflows/check-formatting.yml/badge.svg)
![Docker CI](https://github.com/wenh06/fl-sim/actions/workflows/docker-image.yml/badge.svg?branch=docker-ci)

This repository is migrated from [fl_seminar](https://github.com/wenh06/fl_seminar/tree/master/code)

The main part of this code repository is a standalone simulation framework for federated training.

<!-- toc -->

- [A Simple Simulation Framework for Federated Learning Based on PyTorch](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)
  - [Installation](#installation)
  - [Usage Examples](#usage-examples)
  - [Main Modules](#main-modules)
    - [Nodes](#nodes)
    - [Data Processing](#data-processing)
    - [Optimizers](#optimizers)
    - [Regularizers](#regularizers)
    - [Compression](#compression)
    - [Models](#models)
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

The following code snippet shows how to use the framework to train a model on the `FedProxFEMNIST` dataset using the `FedProx` algorithm.

```python
from fl_sim.data_processing.fedprox_femnist import FedProxFEMNIST
from fl_sim.algorithms.fedprox import (
    FedProxServer,
    FedProxClientConfig,
    FedProxServerConfig,
)

# create a FedProxFEMNIST dataset
dataset = FedProxFEMNIST()
# choose a model
model = dataset.candidate_models["cnn_femmist_tiny"]
# set up the server and client configurations
server_config = FedProxServerConfig(200, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7)
client_config = FedProxClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
# create a FedProxServer object
s = FedProxServer(model, dataset, server_config, client_config)
# normal centralized training
s.train_centralized()
# federated training
s.train_federated()
```

## Main Modules

### [Nodes](fl_sim/nodes.py)

<details>
<summary>Click to expand!</summary>

Nodes are the core of the simulation framework. A node is a PyTorch module that can be trained and evaluated. TODO: more details...

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
