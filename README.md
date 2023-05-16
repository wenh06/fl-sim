# A Simple Simulation Framework for Federated Learning Based on PyTorch

![formatting](https://github.com/wenh06/fl_sim/actions/workflows/check-formatting.yml/badge.svg)
![Docker CI](https://github.com/wenh06/fl_sim/actions/workflows/docker-image.yml/badge.svg?branch=docker-ci)

This repository is migrated from [fl_seminar](https://github.com/wenh06/fl_seminar/tree/master/code)

The main part of this code repository is a standalone simulation framework for federated training.

<!-- toc -->

- [A Simple Simulation Framework for Federated Learning Based on PyTorch](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)
  - [Optimizers](#optimizers)
  - [Regularizers](#regularizers)
  - [Compression](#compression)
  - [Data Processing](#data-processing)
  - [Models](#models)
  - [Algorithms Implemented](#algorithms-implemented)

<!-- tocstop -->

## [Optimizers](fl_sim/optimizers)

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

## [Regularizers](fl_sim/regularizers)

The module (folder) [regularizers](fl_sim/regularizers) contains code for regularizers for model parameters (weights).

1. `L1Norm`
2. `L2Norm`
3. `L2NormSquared`
4. `NullRegularizer`

These regularizers are subclasses of a base class `Regularizer`, and can be obtained by passing the name of the regularizer to the function `get_regularizer`. The regularizers share common methods `eval` and `prox_eval`.

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

## [Compression](fl_sim/compressors)

The module (folder) [compressors](fl_sim/compressors) contains code for constructing compressors.

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

## [Data Processing](fl_sim/data_processing)

The module (folder) [data_processing](fl_sim/data_processing) contains code for data preprocessing, io, etc. The following datasets are included in this module:

1. `FedCIFAR`
2. `FedCIFAR100`
3. `FedEMNIST`
4. `FedShakespeare`
5. `FedSynthetic`
6. `FedProxFEMNIST`
7. `FedProxMNIST`

Additionally, one can get the list of `LIBSVM` datasets via

```python
pd.read_html("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/")[0]
```

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

## [Models](fl_sim/models)

The module (folder) [models](fl_sim/models) contains pre-defined (neural network) models, most of which are very simple:

1. `MLP`
2. `FedPDMLP`
3. `CNNMnist`
4. `CNNFEMnist`
5. `CNNFEMnist_Tiny`
6. `CNNCifar`
7. `RNN_OriginalFedAvg`
8. `RNN_StackOverFlow`
9. `ResNet18`
10. `ResNet10`

One can call the `module_size` or `module_size_` properties to check the size (in terms of number of parameters and memory consumption respectively) of the model.

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)

## [Algorithms Implemented](fl_sim/algorithms)

1. [FedProx](https://github.com/litian96/FedProx) ![test-fedprox](https://github.com/wenh06/fl_sim/actions/workflows/test-fedprox.yml/badge.svg)
2. [FedOpt](https://arxiv.org/abs/2003.00295) ![test-fedopt](https://github.com/wenh06/fl_sim/actions/workflows/test-fedopt.yml/badge.svg)
3. [pFedMe](https://github.com/CharlieDinh/pFedMe) ![test-pfedme](https://github.com/wenh06/fl_sim/actions/workflows/test-pfedme.yml/badge.svg)
4. [FedSplit](https://arxiv.org/abs/2005.05238) ![test-fedsplit](https://github.com/wenh06/fl_sim/actions/workflows/test-fedsplit.yml/badge.svg)
5. [FedDR](https://github.com/unc-optimization/FedDR) ![test-feddr](https://github.com/wenh06/fl_sim/actions/workflows/test-feddr.yml/badge.svg)
6. [FedPD](https://github.com/564612540/FedPD/) ![test-fedpd](https://github.com/wenh06/fl_sim/actions/workflows/test-fedpd.yml/badge.svg)
7. [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html) ![test-scaffold](https://github.com/wenh06/fl_sim/actions/workflows/test-scaffold.yml/badge.svg)
8. [ProxSkip](https://proceedings.mlr.press/v162/mishchenko22b.html) ![test-proxskip](https://github.com/wenh06/fl_sim/actions/workflows/test-proxskip.yml/badge.svg)
9. [Ditto](https://arxiv.org/abs/2012.04221) ![test-ditto](https://github.com/wenh06/fl_sim/actions/workflows/test-ditto.yml/badge.svg)

:point_right: [Back to TOC](#a-simple-simulation-framework-for-federated-learning-based-on-pytorch)
