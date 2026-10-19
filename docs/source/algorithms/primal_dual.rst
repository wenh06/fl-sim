.. _fl_alg_primal_dual:

Primal-Dual Algorithms in Federated Learning
--------------------------------------------

In traditional optimization, primal-dual algorithms are a classic tool for problems with an
explicit primal-dual structure. Consider the consensus optimization problem with an explicit
global-variable constraint (note the subtle difference from the consensus set formulation in the
:ref:`operator splitting <fl_alg_operator_splitting>` and
:ref:`skipping <fl_alg_skipping>` sections),

.. math::
   :label: fedpd-constraint

   \DeclareMathOperator*{\minimize}{minimize}
   \DeclareMathOperator*{\maximize}{maximize}
   \DeclareMathOperator*{\argmin}{arg\,min}
   \newcommand{\col}{\operatorname{col}}
   \newcommand{\R}{\mathbb{R}}

   \begin{array}{cl}
   \minimize & F(\Theta) := \sum\limits_{k=1}^K f_k(\theta_k), \\
   \text{subject to} & \theta_k = \theta, ~ k = 1, \ldots, K,
   \end{array}

with :math:`\Theta = \col(\theta_1, \cdots, \theta_K)` and
:math:`\theta, \theta_1, \ldots, \theta_K \in \R^d`. The *augmented Lagrangian* (AL) of
:eq:`fedpd-constraint` is

.. math::
   :label: al

   \mathcal{L}(\theta, \Theta, \Lambda) = \sum\limits_{k=1}^K \mathcal{L}_k(\theta, \theta_k, \lambda_k),
   \qquad
   \mathcal{L}_k(\theta, \theta_k, \lambda_k) = f_k(\theta_k) + \langle \lambda_k, \theta_k - \theta \rangle + \frac{1}{2s} \lVert \theta_k - \theta \rVert^2,

where :math:`\Lambda = \col(\lambda_1, \ldots, \lambda_K)` collects the *dual variables* (the
Lagrangian multipliers) and :math:`s > 0` is the penalty parameter. Applying a gradient
ascent-type method to the dual problem
:math:`\maximize_\Lambda \ G(\Lambda) := \inf_{\theta, \Theta} \mathcal{L}(\theta, \Theta, \Lambda)`
yields the generic iteration scheme for this class of problems,

.. math::
   :label: al-iter

   \begin{aligned}
   \theta_k^{(t+1)} & = \argmin_{\theta_k} \mathcal{L}_k\left( \theta^{(t)}, \theta_k, \lambda_k^{(t)} \right), \\
   \lambda_k^{(t+1)} & = \lambda_k^{(t)} + \frac{1}{s} \left( \theta_k^{(t+1)} - \theta^{(t)} \right), \\
   \theta^{(t+1)} & = \argmin_{\theta} \sum\limits_{k=1}^K \left\{ \frac{1}{2s} \lVert \theta_k^{(t+1)} - \theta \rVert^2 - \langle \lambda_k^{(t+1)}, \theta \rangle \right\}
   = \frac{1}{K} \sum\limits_{k=1}^K \left( \theta_k^{(t+1)} + s\lambda_k^{(t+1)} \right).
   \end{aligned}

.. _fl_alg_primal_dual_fedpd:

``FedPD``: Federated Primal-Dual Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on this scheme, the ``FedPD`` algorithm [:footcite:ct:`zhang2020fedpd`] alternates the
client-side local updates, the client-side dual updates, and the server-side global averaging.
Two design choices are worth highlighting.

**A flexible local solver.** For the local update in :eq:`al-iter` (the minimization of the
local AL :math:`\mathcal{L}_k`), ``FedPD`` allows an arbitrary client-side solver
:math:`\operatorname{\mathbf{Oracle}}_k` -- plain SGD, or combined with variance-reduction
techniques as in ``SCAFFOLD``. This flexibility is a hallmark of federated optimization (the
same idea as the server-side ``FedOpt``). In the variance-reduced case, one can further
*linearize* :math:`f_k` at the current inner iterate :math:`\theta_k^{(t, r)}` (with local
gradient :math:`g_k^{(t, r)}`), under which the local AL becomes a quadratic whose minimizer has
a closed form,

.. math::
   :label: fedpd-linearize

   \argmin \ \widetilde{\mathcal{L}}(\theta_{k})
   = \frac{s}{s+\gamma} \, \theta_k^{(t,r)} + \frac{\gamma}{s+\gamma} \, \theta_{k, 0}^{(t)} - \frac{s\gamma}{s+\gamma}\left( g_k^{(t,r)} + \lambda_k^{(t)} \right),

where :math:`\gamma` is the linearization (proximal) coefficient and
:math:`\theta_{k, 0}^{(t)}` is the value of the local model at the beginning of the round.

**Skipping the global step.** The outer (server) step in :eq:`al-iter` is executed only with
probability :math:`1 - p`. When it is skipped, the clients use their own
:math:`\widetilde{\theta}^{(t)} = \theta_k^{(t-1)} + s \lambda_k^{(t-1)}` in place of the global
model,

.. math::
   :label: fedpd-iter

   \theta^{(t+1)} = \begin{cases}
   \frac{1}{K} \sum\limits_{k=1}^K \left( \theta_k^{(t+1)} + s\lambda_k^{(t+1)} \right), & \text{with probability } 1 - p, \\
   \theta^{(t)}, & \text{with probability } p,
   \end{cases}

so the expensive server round -- exactly the communication bottleneck -- is skipped with
probability :math:`p`. Moreover, the skipping probability can be made *heterogeneity-aware*:
[:footcite:ct:`zhang2020fedpd`] (TABLE III) relates the achievable :math:`p` to the degree of
data heterogeneity across clients :math:`G` (quantified via the bounded gradient dissimilarity),
the target accuracy :math:`\varepsilon` of the (epsilon-)stationary point, and the Lipschitz
constant :math:`L` of :math:`f, f_k`, with the step size :math:`s = \frac{\sqrt{5}-1}{8L}`:

.. math::
   :label: fedpd-skipping-prob

   p \approx \begin{cases}
   \sqrt{\frac{1}{36s} \cdot \frac{\varepsilon}{G^2}}, & \text{if } 0 \leqslant p < \frac{1-2Ls}{1+Ls}, \\
   1 - 2 \Big/ \log\left(\frac{1}{42s} \cdot \frac{\varepsilon}{G^2}\right), & \text{if } \frac{1-2Ls}{1+Ls} < p < 1.
   \end{cases}

The more homogeneous the client data (smaller :math:`G`), the more aggressively the
communication rounds can be skipped. The pseudocode is summarized below.

.. _pcode-fedpd:

.. include:: ../_algo_pcode/fedpd.rst

.. footbibliography::
