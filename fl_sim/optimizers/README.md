# Optimizers that solve **inner (local)** optimization problems

Some of the researchers let the `Optimizers` maintain some intermediate variables,
for example, the `weights` in the proximal term, the `gradient cache` for variance reduction.
However, we let the `Algorithms` maintain such variables, and make them input parameters for relevant `Optimizers`.

Most (inner) optimizers are based on the [ProxSGD](base.py) optimizer, including

- [FedDROptimizer](feddr.py)
- [FedProxOptimizer](fedprox.py)
- [pFedMeOptimizer](pfedme.py)

Other (inner) optimizers include

- [FedPD Optimizers](fedpd.py) (Not checked yet)
