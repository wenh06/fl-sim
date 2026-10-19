.. _fl_alg_pfl:

Personalized Federated Learning Algorithms
---------------------------------------------------

In personalized federated learning (PFL), each client seeks a model tailored to its own data
distribution while still benefiting from the knowledge of the other clients. A simple and general
way to formulate PFL is via *weak consensus*: instead of enforcing full consensus
:math:`\theta_1 = \cdots = \theta_K`, one adds a regularizer :math:`\mathcal{R}` that penalizes the
deviation between the local models. This page collects the PFL algorithms (implemented in
``fl_sim`` or closely related) together with their modeling, theory, and pseudocodes.

.. _fl_alg_pfl_l2sgd:

``L2SGD``: Loopless Local SGD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest algorithm of this family is the loopless stochastic gradient descent
(Loopless Local SGD, ``L2SGD``) introduced in [:footcite:ct:`hanzely2020federated`].
It considers the following weak-consensus problem

.. math::
   :label: l2sgd

   \begin{array}{cl}
   \minimize & F(\Theta) = f(\Theta) + \lambda \varphi(\Theta) \\
   \text{where} & f(\Theta) = \frac{1}{K} \sum\limits_{k=1}^K f_k(\theta_k), \\
   & \varphi(\Theta) = \frac{1}{2K} \sum\limits_{k=1}^K \left\lVert \theta_k - \bar{\theta} \right\rVert^2
   = \frac{1}{2K} \sum\limits_{k=1}^K \left\lVert \theta_k - \frac{1}{K} \sum\limits_{j=1}^K \theta_j \right\rVert^2, \\
   & \Theta = \col(\theta_1, \ldots, \theta_K),
   \end{array}

where :math:`\lambda \geqslant 0` is the penalty coefficient and
:math:`\bar{\theta} = \frac{1}{K} \sum_{k=1}^K \theta_k` is the global average model.
For :math:`0 < \lambda < \infty`, the solution :math:`\bar{\theta}, \theta_1, \ldots, \theta_K` is
called a set of *mixed models*. It is easy to see that this problem is equivalent to the problem
considered by ``FedProx`` (cf. the constrained problem in the
:ref:`proximal algorithms section <fl_alg_proximal>`).

Instead of evaluating the full gradient :math:`\nabla F = \nabla f + \lambda \nabla \varphi` at
every iteration, [:footcite:ct:`hanzely2020federated`] defines an unbiased stochastic gradient of
:math:`F`

.. math::
   :label: l2sgd-grad

   G(\Theta) := \begin{cases}
   \frac{\nabla f(\Theta)}{1 - p}, & \text{with probability } 1 - p, \\
   \frac{\lambda \nabla \varphi(\Theta)}{p}, & \text{with probability } p,
   \end{cases}

with :math:`p \in (0, 1)`. Note that computing
:math:`\nabla f(\Theta) = \col(\nabla f_1(\theta_1), \ldots, \nabla f_K(\theta_K))` is completely
block-separable: in the federated setting, this step is carried out locally on the clients without
any communication with the server, which is a significant advantage in communication-constrained
scenarios. In this sense, ``L2SGD`` already incorporates the *loopless* (skipping) idea; cf. the
``ProxSkip`` algorithm in the :ref:`skipping algorithms section <fl_alg_skipping>`. Under the
additional finite-sum assumption :math:`f_k(\theta_k) = \sum_{j=1}^m f_{k,j}(\theta_k)`, one can
further combine the stochastic gradient with randomized Kaczmarz-type (importance-sampled) updates;
see [:footcite:ct:`Kovalev2020_loopless`] for the analysis of loopless methods.
The resulting method is summarized below.

.. _pcode-l2sgd:

.. include:: ../_algo_pcode/l2sgd.rst

.. _fl_alg_pfl_pfedme:

``pFedMe``: Personalized Federated Mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``pFedMe`` algorithm [:footcite:ct:`t2020pfedme`] considers a weak-consensus problem of the
same family as :eq:`l2sgd`, with the global model appearing as a single variable
:math:`\theta \in \R^d`,

.. math::
   :label: pfedme-obj

   \DeclareMathOperator*{\minimize}{minimize}
   \DeclareMathOperator*{\argmin}{arg\,min}
   \newcommand{\R}{\mathbb{R}}
   \newcommand{\prox}{\mathbf{prox}}

   \minimize_{\theta, \theta_1, \ldots, \theta_K} \quad \sum\limits_{k=1}^K \left\{ f_k(\theta_k) + \frac{\lambda}{2} \left\lVert \theta_k - \theta \right\rVert^2 \right\},

where :math:`\lambda \geqslant 0` is the penalty coefficient. Exploiting the center--client
structure of federated learning, the problem is equivalently reformulated as a *bi-level
optimization problem*,

.. math::
   :label: pfedme-bilevel

   \begin{array}{cl}
   \minimize & F(\theta) := \frac{1}{K} \sum\limits_{k=1}^K F_k(\theta), \\
   \text{where} & F_k(\theta) = \min\limits_{\theta_k \in \R^d} \left\{ f_k(\theta_k) + \frac{\lambda}{2} \left\lVert \theta_k - \theta \right\rVert^2 \right\},
   \end{array}

whose inner objective :math:`F_k(\theta)` is exactly the *Moreau envelope* of :math:`f_k`,
:math:`\mathcal{M}_{f_k, \lambda}(\theta) := \inf_{\theta_k} \{ f_k(\theta_k) + \frac{\lambda}{2} \lVert \theta_k - \theta \rVert^2 \}`,
and the inner minimizer is the proximal operator
:math:`\prox_{f_k, \lambda}(\theta) := \argmin_{\theta_k} \{ f_k(\theta_k) + \frac{\lambda}{2} \lVert \theta_k - \theta \rVert^2 \}`
evaluated at the global model. This proximal point is precisely the personalized model of client
:math:`k`. By the Moreau envelope--proximal operator relation, the gradient of the outer
objective is

.. math::
   :label: pfedme-grad

   \nabla F(\theta) = \frac{1}{K} \sum\limits_{k=1}^K \nabla \mathcal{M}_{f_k, \lambda} (\theta)
   = \frac{\lambda}{K} \sum\limits_{k=1}^K \left( \theta - \prox_{f_k, \lambda} (\theta) \right),

so that one (stochastic) gradient step on :math:`F` decomposes into the average of the client-side
partial updates :math:`\theta^{(t)} - \eta \lambda \left( \theta^{(t)} - \prox_{f_k, \lambda} (\theta^{(t)}) \right)`.
Operationally, each client holds a copy :math:`\omega_k^{(t, r)}` of the global model and,
between two communications, alternates for :math:`R` rounds

.. math::
   :label: pfedme-local

   \theta_k^{(t, r)} \gets \prox_{f_k, \lambda} \left( \omega_k^{(t, r)}; \ b_r \right),
   \qquad
   \omega_k^{(t, r+1)} \gets \omega_k^{(t, r)} - \eta \lambda \left( \omega_k^{(t, r)} - \prox_{f_k, \lambda} \left( \omega_k^{(t, r)} \right) \right),

where :math:`b_r` is the minibatch of round :math:`r`; the server then aggregates

.. math::
   :label: pfedme-agg

   \theta^{(t+1)} \gets (1-\beta)\,\theta^{(t)} + \frac{\beta}{\# \mathcal{S}^{(t)}} \sum\limits_{k \in \mathcal{S}^{(t)}} \omega_{k}^{(t, R)}.

Note that the two parts of :eq:`pfedme-local` are executed for :math:`R` local rounds *before* any
communication happens -- yet another instance of the loopless (skipping) idea;
cf. :ref:`L2SGD <fl_alg_pfl_l2sgd>` and the :ref:`skipping algorithms section <fl_alg_skipping>`.
The pseudocode is summarized below.

.. _pcode-pfedme:

.. include:: ../_algo_pcode/pfedme.rst

.. _fl_alg_pfl_ditto:

``Ditto``: Fair and Robust Personalized Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Ditto`` algorithm [:footcite:ct:`li_2021_ditto`] develops the proximal-subproblem idea of
``FedProx`` into a *bi-level* formulation seen from the *client's* perspective (``APFL``,
proposed slightly earlier, adopts a similar viewpoint; see below),

.. math::
   :label: ditto-bilevel

   \begin{array}{cl}
   \minimize\limits_{\omega_k} & h_k(\omega_k, \theta^*) := f_k(\omega_k) + \frac{\mu}{2} \lVert \omega_k - \theta^* \rVert^2, \\
   \text{subject to} & \theta^* \in \argmin\limits_{\theta} G\left( f_1(\theta), \cdots, f_K(\theta) \right),
   \end{array}

where :math:`G` is the aggregation rule applied by the server to the client models, e.g. plain
averaging as in ``FedAvg``. Reading the formulation: the inner problem is a plain federated
optimization problem on the global model :math:`\theta` alone, while the outer problem attaches
to each client a proximal task for personalization -- and the personalized model
:math:`\omega_k` does *not* participate in the global update. The global and the personalized
models are thereby decoupled, and their updates can be scheduled flexibly.

Consequently, each client solves two problems per round: the global update
:math:`\theta_k^{(t)} \gets \operatorname{\mathbf{UpdateGlobal}}(\theta^{(t)}, f_k)` (the inner
problem), and the proximal task whose solution tracks the moving center
:math:`\theta^{(t)}`,

.. math::
   :label: ditto-local

   \omega_k^{(t+1)} \gets \omega_k^{(t)} - \eta \left( \nabla f_k(\omega_k^{(t)}) + \mu \left( \omega_k^{(t)} - \theta^{(t)} \right) \right).

Unlike ``FedSplit`` or ``pFedMe``, where the proximal operator is solved in an inner loop to
(approximate) optimality, ``Ditto`` executes only a *single* (stochastic) gradient step per round
for the outer problem, letting the proximal center :math:`\theta^{(t)}` evolve alongside the
global model. The pseudocode is summarized below.

.. _pcode-ditto:

.. include:: ../_algo_pcode/ditto.rst

.. _fl_alg_pfl_apfl:

``APFL``: Adaptive Personalized Federated Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Proposed slightly before ``Ditto`` [:footcite:ct:`deng2020_apfl`], ``APFL`` considers the
bi-level problem

.. math::
   :label: apfl-bilevel

   \begin{array}{cl}
   \minimize\limits_{\omega_k \in \mathrm{dom} f_k} & h_k(\omega_k, \theta^*) := f_k\left( \alpha_k \omega_k + (1 - \alpha_k) \theta^* \right), \\
   \text{subject to} & \theta^* \in \argmin\limits_{\theta} \frac{1}{K} \sum\limits_{k=1}^K f_k(\theta),
   \end{array}

with the *mixture weight* :math:`\alpha_k \in (0, 1)`. The main difference from
:eq:`ditto-bilevel` is the personalization mechanism: instead of a quadratic proximal term, the
local objective is evaluated at the convex combination
:math:`\alpha_k \omega_k + (1 - \alpha_k) \theta^*` of an auxiliary local model and the global
model, and :math:`\alpha_k` plays the same "personalization degree" role as the penalty
:math:`\mu` in a proximal term. The resulting personalized model is

.. math::
   :label: apfl-model

   \hat{\omega}_k = \alpha_k \omega_k^* + (1 - \alpha_k) \theta^*,
   \qquad
   \omega_k^* = \argmin\limits_{\omega_k \in \mathrm{dom} f_k} f_k\left( \alpha_k \omega_k + (1 - \alpha_k) \theta^* \right).

A caveat worth noting: when :math:`\mathrm{dom} f_k` is a *proper* subset of :math:`\R^d`, the
convex combination in :eq:`apfl-model` may fail to attain a minimizer of :math:`f_k` -- the range
of :math:`\alpha_k \omega_k^* + (1 - \alpha_k) \theta^*` can be a strict subset of the search
domain that excludes the minimizer. Enlarging the search domain of :math:`\omega_k` restores
attainability, but then :math:`\hat{\omega}_k` could equally be obtained by minimizing
:math:`f_k` alone, i.e. by pure local training -- which contradicts the very goal of personalized
federated learning. The pseudocode is summarized below.

.. _pcode-apfl:

.. include:: ../_algo_pcode/apfl.rst

.. _fl_alg_pfl_feddyn:

``FedDyn``: Dynamic Regularized Federated Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithms above (and ``FedProx`` alike) regularize the local objective with a *static*
proximal term: in round :math:`(t+1)`, client :math:`k` minimizes
:math:`f_k(\theta_k) + \frac{\mu}{2} \lVert \theta_k - \theta^{(t)} \rVert^2`.
The ``FedDyn`` algorithm [:footcite:ct:`acar2021feddyn`] refines this into a *dynamically
regularized* objective,

.. math::
   :label: feddyn-obj

   F_k(\theta_k) := f_k(\theta_k) - \left\langle \nabla f_k\left( \theta_k^{(t)} \right), \theta_k \right\rangle + \frac{\mu}{2} \lVert \theta_k - \theta^{(t)} \rVert^2,

which is equivalent to

.. math::
   :label: feddyn-adjusted

   F_k(\theta_k) = \left( f_k(\theta_k) + \frac{\mu}{2} \lVert \theta_k - \theta^{(t)} \rVert^2 \right)
   - \left\langle \nabla f_k\left( \theta_k^{(t)} \right), \theta_k - \theta_k^{(t)} \right\rangle.

In words: each (stochastic) step on :math:`F_k` follows the descent direction of the
prox-augmented objective, *corrected* by the linearization of :math:`f_k` at the previous local
iterate :math:`\theta_k^{(t)}` (with :math:`\mathfrak{g}_k^{(t)} := \nabla f_k(\theta_k^{(t)})`).
Another remarkable property is the first-order optimality condition of :eq:`feddyn-obj`,

.. math::
   :label: feddyn-opt

   \nabla f_k\left( \theta_k^{(t+1)} \right) - \nabla f_k\left( \theta_k^{(t)} \right) + \mu \left( \theta_k^{(t+1)} - \theta^{(t)} \right) = 0,

which yields the recursion

.. math::
   :label: feddyn-grad-iter

   \mathfrak{g}_k^{(t+1)} \gets \mathfrak{g}_k^{(t)} - \mu \left( \theta_k^{(t+1)} - \theta^{(t)} \right),

i.e. the local gradient estimate is maintained *without* additional function or gradient
evaluations -- a substantial computational and implementational convenience. The pseudocode is
summarized below.

.. _pcode-feddyn:

.. include:: ../_algo_pcode/feddyn.rst

.. _fl_alg_pfl_pfedmac:

``pFedMac``: Personalized FL via Maximizing Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Related to the "direction correction" of ``FedDyn``, the ``pFedMac`` algorithm
[:footcite:ct:`li2021pfedmac`] considers the bi-level problem

.. math::
   :label: pfedmac-obj

   \begin{array}{cl}
   \minimize & F(\theta) = \frac{1}{K} \sum\limits_{k=1}^K F_k(\theta), \\
   \text{where} & F_k(\theta) := \min\limits_{\theta_k} \left\{ f_k(\theta_k) - \mu \langle \theta_k, \theta \rangle + \frac{\mu}{2} \lVert \theta \rVert^2_2 \right\},
   \end{array}

in which the inner problem corrects the personalized model :math:`\theta_k` using the global
model :math:`\theta` through their inner product. In probability and statistics,
:math:`\langle \theta_k, \theta \rangle` quantifies the *correlation* between the personalized
and the global models; hence the technique is known as *maximizing correlation*. Two elementary
rewritings connect it to the formulations above: moving the global term out of the inner
minimization gives :math:`F_k(\theta) = F_k'(\theta) + \frac{\mu}{2} \lVert \theta \rVert^2_2`
with :math:`F_k'(\theta) := \min_{\theta_k} \{ f_k(\theta_k) - \mu \langle \theta_k, \theta \rangle \}`,
and completing the square shows

.. math::
   :label: pfedmac-adjusted

   F_k(\theta) = \min\limits_{\theta_k} \left\{ f_k(\theta_k) + \frac{\mu}{2} \lVert \theta_k - \theta \rVert^2_2 - \frac{\mu}{2} \lVert \theta_k \rVert^2_2 \right\},

i.e. the prox-augmented objective corrected by a quadratic term in :math:`\theta_k` itself --
exactly the flavor of :eq:`feddyn-adjusted`, so the same one-step direction-correction updates
apply. ``pFedMac`` further studies *sparse* personalized models via an additional
:math:`\ell_1` term,

.. math::
   :label: pfedmac-sparse

   F_k(\theta) := \min\limits_{\theta_k} \left\{ f_k(\theta_k) - \mu \langle \theta_k, \theta \rangle + \frac{\mu}{2} \lVert \theta \rVert^2_2 + \gamma \lVert \theta_k \rVert_1 \right\},

and extends the center--client hierarchy to a *three-level* structure with sub-centers (edges or
hubs), each aggregating a cluster :math:`\{ \theta_{k,1}, \ldots, \theta_{k, J_k} \}` of clients
before communicating with the center -- a structure expected to become increasingly common as
federated networks scale, and closely related to clustered FL. The pseudocode is summarized
below.

.. _pcode-pfedmac:

.. include:: ../_algo_pcode/pfedmac.rst

.. _fl_alg_pfl_fpfc:

``FPFC``: Fusion Penalized Federated Clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond regularizing the deviation between each pair of *local* models (as in ``L2SGD``) or between
the local model and the global one, one can also encourage *clusters* of clients to share identical
models, without knowing the clusters a priori. The fusion penalized federated clustering
(``FPFC``) algorithm (see `arXiv:2211.04218 <https://arxiv.org/abs/2211.04218>`_) achieves this by
imposing a nonconvex *pairwise* fusion penalty on the model differences. Introducing the auxiliary
variables :math:`\theta_{ij} \equiv \omega_i - \omega_j` for all pairs :math:`i, j \in [m]`, it considers

.. math::
   :label: fpfc-obj

   \begin{array}{cl}
   \minimize\limits_{\omega, \theta} & \sum\limits_{i=1}^m f_i(\omega_i) + \frac{1}{2m} \sum\limits_{i=1}^m \sum\limits_{j=1}^m \tilde{g}(\lVert \theta_{ij} \rVert) \\
   \text{subject to} & \omega_i - \omega_j = \theta_{ij}, \qquad i, j \in [m],
   \end{array}

where :math:`\tilde{g}(\cdot)` is a smoothed SCAD-type penalty (parameterized by
:math:`\xi, \lambda, a`) whose proximal operator shrinks small pairwise differences towards zero
(clients in the same cluster) while keeping large ones (clients in different clusters) almost
untouched. The problem is solved via an ADMM-style (Douglas--Rachford) splitting with the augmented
Lagrangian penalty :math:`\rho` and dual variables :math:`v_{ij}`: each round, a random subset of
devices performs :math:`T_i` local gradient epochs on
:math:`f_i(\omega) + \frac{\rho}{2} \lVert \omega - \zeta_i \rVert^2`, and the server applies the
piecewise proximal update to :math:`\theta_{ij}` and ascends the duals for the pairs of active
devices. After training, devices :math:`i` and :math:`j` are assigned to the same cluster whenever
:math:`\lVert \theta_{ij} \rVert` is (nearly) zero. The pseudocode is summarized below; note that
neither the number of clusters nor the cluster membership needs to be specified in advance.

.. _pcode-fpfc:

.. include:: ../_algo_pcode/fpfc.rst

.. _fl_alg_pfl_fedcr:

``FedCR``: Personalized Federated Learning Based on Across-Client Common Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The methods above personalize the *whole* local model. ``FedCR`` (see
`ICML 2023 <https://proceedings.mlr.press/v202/zhang23w.html>`_) instead decouples each local model
:math:`w_i = [w_i^f, w_i^p]` into a *feature extractor* (body) :math:`w_i^f`, shared across
clients and aggregated at the server, and a *predictor* (head) :math:`w_i^p`, kept personal. The
extractors are aligned through a *common representation*: for each class :math:`c`, the local
class-conditional mutual information is constrained to be close to the global one, which (via the
Markov chain :math:`y \to x \to z`) amounts to a class-wise feature alignment

.. math::
   :label: fedcr-cmi

   I(z; x \mid y_i) - I_i(z; x_i \mid y_i) = \mathbb{E}_{p(x_i, y_i)} \mathbb{E}_{p(x \mid x_i, y_i)} \left[ \mathrm{KL}\left[ p(z \mid x) \,\middle\|\, p(z \mid x_i) \right] \right],

weighted by a Lagrange multiplier :math:`\beta \geqslant 0` (the CMI regularizer). The
representation mapping is modeled as a diagonal Gaussian
:math:`p(z \mid x_i) = \mathcal{N}\left(z \,\middle|\, \mu(x_i), \Sigma(x_i)\right)`, under which
the intractable global posterior is estimated by a *product of experts* (PoE) over the class-wise
local posteriors of the active clients :math:`\mathcal{P}_t`, with prior :math:`p(z) = \mathcal{N}(0, 1)`:

.. math::
   :label: fedcr-poe

   p(z^c \mid x) \propto p(z) \prod_{i \in \mathcal{P}_t} p^c(z \mid x_i), \qquad
   p^c(z \mid x_i) = \prod_{n = 1}^{N_i^{c}} p\left(z \,\middle|\, x_i^{(n)}\right)^{1 / N_i^{c}},

where :math:`p^c(z \mid x_i)` is the (geometric-mean) local PoE over the :math:`N_i^{c}` samples
of class :math:`c` at client :math:`i`, computed on the fly from the latest forward passes of
local training. As a product of Gaussians is itself Gaussian, both PoEs reduce to class-wise
means and covariances: each client uploads :math:`(\mu_i^c, \Sigma_i^c)` for its classes only,
and the server combines them analytically via the precisions,

.. math::
   :label: fedcr-poe-closed

   \Sigma^c = \left( I + \sum_{i \in \mathcal{P}_t} \left( \Sigma_i^c \right)^{-1} \right)^{-1}, \qquad
   \mu^c = \Sigma^c \sum_{i \in \mathcal{P}_t} \mu_i^c \left( \Sigma_i^c \right)^{-1},

besides averaging the extractors, :math:`w^f = \frac{1}{\lvert \mathcal{P}_t \rvert} \sum_{i \in \mathcal{P}_t} w_i^f`.
Classes missed by all active clients keep the statistics of the previous round. The local
objective of client :math:`i` is its empirical loss plus the alignment term above,

.. math::
   :label: fedcr-obj

   L_i = f_i(w_i) + \beta \, \mathbb{E}_{p(x_i, y_i)} \left[ \mathrm{KL}\left[ p(z^{c = y_i} \mid x) \,\middle\|\, p(z \mid x_i) \right] \right],

optimized with the re-parameterization trick for the stochastic features. Since the PoE posterior
can be sharper than any single expert, the common representation carries not only the common but
also the complementary information across clients; the alignment term doubles as noise injection
into the local features, which improves generalization and uncertainty calibration. Note that for
:math:`\beta = 0` and :math:`\Sigma(x_i) = 0` (deterministic features), the method degenerates to
``FedPer``. The pseudocode is summarized below.

.. _pcode-fedcr:

.. include:: ../_algo_pcode/fedcr.rst

.. footbibliography::
