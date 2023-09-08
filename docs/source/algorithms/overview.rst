.. _fl_alg_overview:

Overview of Optimization Algorithms in Federated Learning
---------------------------------------------------------

The most important contribution of the initial work on federated learning [:footcite:ct:`mcmahan2017fed_avg`]
was the introduction of the Federated Averaging algorithm (``FedAvg``). Mathematically, federated learning
solves the following problem of minimization of empirical risk function

.. math::
   :label: fl-basic-dist

   \DeclareMathOperator*{\expectation}{\mathbb{E}}
   \DeclareMathOperator*{\minimize}{minimize}
   \newcommand{\R}{\mathbb{R}}

   \begin{array}{cl}
   \minimize\limits_{\theta \in \R^d} & f(\theta) = \expectation\limits_{k \sim {\mathcal{P}}} [f_k(\theta)], \\
   \text{where} & f_k(\theta) = \expectation\limits_{(x, y) \sim \mathcal{D}_k} [\ell_k(\theta; x, y)],
   \end{array}

where :math:`\ell_k` is the loss function of client :math:`k`,
:math:`\mathcal{D}_k` is the data distribution of client :math:`k`,
:math:`\mathcal{P}` is the distribution of clients, and :math:`\mathbb{E}` is the expectation operator.
If we simply let :math:`\mathcal{P} = \{1, 2, \ldots, K\}`, then the optimization problem can be simplified as

.. math::
   :label: fl-basic

   \begin{array}{cl}
   \minimize\limits_{\theta \in \R^d} & f(\theta) = \sum\limits_{k=1}^K w_k f_k(\theta).
   \end{array}

For further simplicity, we often take :math:`w_k = \frac{1}{K}`. The functions :math:`f_k` and :math:`f`
are usually assumed to satisfy the following conditions:

   * (A1) :math:`f_k` and :math:`f` are :math:`L`-smooth (:math:`L > 0`), i.e. they have :math:`L`-Lipschitz continuous gradients:
   
      .. math::
         :label: l-smooth

         \begin{array}{c}
         \lVert \nabla f (\theta) - f (\theta') \rVert \leqslant L \lVert \theta - \theta' \rVert, \\
         \lVert \nabla f_k (\theta) - f_k (\theta') \rVert \leqslant L \lVert \theta - \theta' \rVert,
         \end{array}
         \quad \forall \theta, \theta' \in \R^d, k = 1, \ldots, K.
   * (A2) The range of :math:`f`

      .. math::

         \DeclareMathOperator*{\dom}{dom}
         
         \dom(f) := \{ \theta \in \R^d ~|~ f(\theta) < + \infty \}

      is nonempty and lower bounded, i.e. there exists a constant :math:`c \in \R` such that

      .. math::
         :label: lower-bounded

         f(\theta) \geqslant c > -\infty, ~ \forall \theta \in \R^d,

      or equivalently,
      
         .. math::
            :label: lower-bounded-2

            f^* := \inf\limits_{\theta \in \R^d} f(\theta) > - \infty.

In many cases, in order to facilitate the analysis of convergence, we will also make some assumptions about
the gradient of the objective function:

   * (A3) Bounded gradient: there exists a constant :math:`G > 0` such that

      .. math::
         :label: bdd_grad

         \lVert \nabla f_k (\theta) \rVert^2 \leqslant G^2, ~ \forall \theta \in \R^d, ~ k = 1, \ldots K.

And the following assumptions on data distributions:

   * (A4-1) Data distribution is I.I.D. (identically and independently distributed) across clients, i.e.
   
      .. math::
         :label: iid-1

         \nabla f(\theta) = \expectation [f_k(\theta)] = \expectation\limits_{(x, y) \sim \mathcal{D}_k}[\nabla \ell_k(\theta; x, y)], ~ \forall \theta \in \R^d, ~ k = 1, \ldots K,

      or equivalently, for any :math:`\varepsilon > 0`, there exists a constant :math:`B \geqslant 0` such that

      .. math::
         :label: iid-2

         \sum\limits_{k=1}^K \lVert \nabla f_k(\theta) \rVert^2 = \lVert f(\theta) \rVert^2,
         ~ \forall \theta \in \left\{ \theta \in \R^d ~ \middle| ~ \lVert f(\theta) \rVert^2 > \varepsilon \right\}.

.. _bdd_grad_dissim:

   * (A4-2) Data distribution is non-I.I.D across clients, in which case we need a quantity to measure
      the degree of this statistical heterogeneity. This quantity can be defined in a number of ways
      [:footcite:ct:`karimireddy2020scaffold, zhang2020fedpd, li2019convergence, sahu2018fedprox`].
      For example, in [:footcite:ct:`karimireddy2020scaffold`] and [:footcite:ct:`zhang2020fedpd`],
      the so-called bounded gradient dissimilarity (BGD), denoted as :math:`(G; B)`-BGD, is used as this quantity.
      More specifically, there exists constants :math:`G > 0` and :math:`B \geqslant 0` such that

      .. math::
         :label: bdd_grad_dissim

         \dfrac{1}{K} \sum\limits_{k=1}^K \lVert \nabla f_k(\theta) \rVert^2 \leqslant G^2 + B^2 \lVert \nabla f(\theta) \rVert^2, ~ \forall \theta \in \R^d.

      It should be noted that letting :math:`B = 0`, the bounded gradient dissimilarity condition (A4-2) degenrates
      to the bounded gradient condition (A3).

Sometimes, in the proof of algorithm convergence, one needs to make assumptions on the convexity of the
objective function :math:`f`, which can be defined as follows:

   * (A5-1) convexity:

      .. math::
         :label: def-convex-function

         f(a \theta + (1 - a) \theta') \leqslant a f(\theta) + (1 - a) f(\theta'),
         ~ \forall \theta, \theta' \in \mathcal{C}, ~ \alpha \in [0, 1].

      where :math:`\mathcal{C}` is a convex set on which :math:`f` is defined.
   * (A5-2) Strong convexity: there exists a constant :math:`\mu > 0` such that :math:`f - \frac{\mu}{2} \lVert \theta \rVert^2`
      is convex. In this case, we say that :math:`f` is :math:`\mu`-strongly convex.

Due to the natural layered and decoupled structure of the federal learning problem, it is more natural to consider
the following constrained optimization problem:

.. math::
   :label: fl-basic-constraint

   \begin{array}{cl}
   \minimize & \frac{1}{K} \sum\limits_{k=1}^K f_k(\theta_k), \\
   \text{subject to} & \theta_k = \theta, ~ k = 1, \ldots, K.
   \end{array}

It is easy to find the equivalence between the constrained optimization problem :eq:`fl-basic-constraint`
and the unconstrained optimization problem :eq:`fl-basic`. The constrained formulation
:eq:`fl-basic-constraint` is called the **consensus problem** in the literature of distributed optimization [:footcite:ct:`boyd2011distributed`]. The superiority of the constrained formulation :eq:`fl-basic-constraint` is that
the objective function becomes block-separable, which is more suitable for the design of parallel and distributed algorithms.

Federated Averaging Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core idea of the ``FedAvg`` algorithm is to make full use of the local computation resources of each client
so that each client can perform multiple local iterations before uploading the local model to the server.
It alleviates the problem of straggler clients and reduces the communication overhead,
hence accelerating the convergence of the algorithm. This may well be thought of as a simple form of
**skipping** algorithm, which were further developed in [:footcite:ct:`zhang2020fedpd, proxskip, proxskip-vr`].
Pseudocode for ``FedAvg`` is shown as follows:

.. _pseduocode-fedavg:

.. image:: ../generated/algorithms/fedavg.svg
   :align: center
   :width: 80%
   :alt: Psuedocode for ``FedAvg``
   :class: no-scaled-link

``FedAvg`` achieved some good numerical results (see Section 3 of [:footcite:ct:`mcmahan2017fed_avg`]),
but its convergence, espcially under non-I.I.D. data distributions, is not properly analyzed
(see [:footcite:ct:`khaled2019_first, Khaled2020_tighter`]). There are several works that deal with this issue
(such as [:footcite:ct:`zhou_2018_convergence, li2019convergence`]) with extra assumptions such as
the convexity of the objective function :math:`f`, etc.

``FedAvg`` from the Perspective of Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, we will analyze the ``FedAvg`` algorithm from the perspective of optimization theory.
In fact, the optimization problem :eq:`fl-basic` that ``FedAvg`` solves can be equivalently reformulated
as the following constrained optimization problem:

.. math::
   :label: fedavg-constraint

   \newcommand{\col}{\operatorname{col}}

   \begin{array}{cl}
   \minimize & F(\Theta) := \frac{1}{K} \sum\limits_{k=1}^K f_k(\theta_k), \\
   \text{subject to} & \Theta \in \mathcal{E},
   \end{array}

where :math:`\Theta = \col(\theta_1, \cdots, \theta_K) := \begin{pmatrix} \theta_1 \\ \vdots \\ \theta_K \end{pmatrix}, \theta_1, \ldots, \theta_K \in \R^d`
and :math:`\mathcal{E} = \left\{ \Theta ~ \middle| ~ \theta_1 = \cdots = \theta_K \right\}` is a convex set in :math:`\R^{Kd}`.
Projected gradient descent (PGD) is an effective method for solving the constrained optimization problem :eq:`fedavg-constraint`,
which has the following update rule:

.. math::
   :label: fedavg-pgd

   \Theta^{(t+1)} = \Pi_{\mathcal{E}} \left( \Theta^{(t)} - \eta \nabla F(\Theta^{(t)}) \right),

where :math:`\Pi_{\mathcal{E}}` is the projection operator onto the set :math:`\mathcal{E}`. It is easy to show that
the projection operator onto the set :math:`\mathcal{E}` is indeed the average operator, i.e.,

.. math::
   :label: fedavg-projection

   \Pi_{\mathcal{E}}: \R^{Kd} \to \mathcal{E}: ( \theta_1, \ldots, \theta_K) \mapsto \left(\frac{1}{K}\sum\limits_{k=1}^K \theta_K, \ldots, \frac{1}{K}\sum\limits_{k=1}^K \theta_K \right).

We have shown that mathematically the ``FedAvg`` algorithm is indeed a kind of stochastic projected gradient descent (SPGD)
algorithm, where the clients perform local stochastic gradient descent (SGD) updates and the server performs
the projection step :eq:`fedavg-projection`.

A Direct Improvement of ``FedAvg``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since ``FedAvg`` is based on stochastic gradient descent (SGD), it is natural to consider applying
acceleration techniques [:footcite:ct:`adagrad, adam, Zaheer_2018_yogi, adamw_amsgrad`] to improve the algorithm performance.
Computation on clients and on the server are relatively decoupled, so it does not require large modifications
to the whole algorithm framework. Indeed, the authors of the ``FedAvg`` paper put this idea into practice and proposed
a federated learning framework called ``FedOpt`` [:footcite:ct:`reddi2020fed_opt`] which has stronger adaptability.
The pseudocode for ``FedOpt`` is shown as follows:

.. _pseduocode-fedopt:

.. image:: ../generated/algorithms/fedopt.svg
   :align: center
   :width: 80%
   :alt: Psuedocode for ``FedOpt``
   :class: no-scaled-link

In the above pseudocode, :math:`\operatorname{aggregate} \left( \left\{ \Delta_{k}^{(t)} \right\}_{k \in \mathcal{S}^{(t)}} \right)`
refers to some method that aggregates the local inertia updates :math:`\Delta_{k}^{(t)}` from the selected clients
:math:`\mathcal{S}^{(t)}` into a global inertia update :math:`\Delta^{(t)}`. This method, for example, can be simply averaging

.. math::
   :label: fedopt-agg-inertia-average

   \Delta^{(t)} \gets \frac{1}{\lvert \mathcal{S}^{(t)} \rvert} \sum\limits_{k \in \mathcal{S}^{(t)}} \Delta_{k}^{(t)},

or linear combination with inertia of the previous iteration

.. math::
   :label: fedopt-agg-inertia-lin-comb

   \Delta^{(t)} \gets \beta_1 \Delta^{(t-1)} + \left( (1 - \beta_1) / \lvert \mathcal{S}^{(t)} \rvert \right) \sum_{k \in \mathcal{S}^{(t)}} \Delta_{k}^{(t)}.

As one has already noticed, compared to ``FedAvg``, ``FedOpt`` introduces some momentum terms on the server node (in **ServerOpt**) to
accelerate the convergence. In [:footcite:ct:`reddi2020fed_opt`], the authors listed several options for **ServerOpt**:

- ``FedAdagrad``:

   .. math::
      :label: fedopt-serveropt-fedadagrad
   
      \begin{aligned}
      v^{(t)} & \gets v^{(t-1)} + ( \Delta^{(t)} )^2 \\
      \theta^{(t+1)} & \gets \theta^{(t)} + \eta_g \Delta^{(t)} / (\sqrt{v^{(t)}}+\tau)
      \end{aligned}

- ``FedYogi``:

   .. math::
      :label: fedopt-serveropt-fedyogi

      \begin{aligned}
      v^{(t)} & \gets v^{(t-1)} - (1 - \beta_2) ( \Delta^{(t)} )^2 \operatorname{sign}(v^{(t-1)} - ( \Delta^{(t)} )^2) \\
      \theta^{(t+1)} & \gets \theta^{(t)} + \eta_g \Delta^{(t)} / (\sqrt{v^{(t)}}+\tau)
      \end{aligned}

- ``FedAdam``:

   .. math::
      :label: fedopt-serveropt-fedadam

      \begin{aligned}
      v^{(t)} & \gets \beta_2 v^{(t-1)} + (1 - \beta_2) ( \Delta^{(t)} )^2 \\
      \theta^{(t+1)} & \gets \theta^{(t)} + \eta_g \Delta^{(t)} / (\sqrt{v^{(t)}}+\tau)
      \end{aligned}

``FedOpt`` applys acceleration techniques which are frequently used in general machine learning tasks to the field of
federated learning. It is a direct improvement of ``FedAvg`` which is simple but important. Moreover, it demonstrates
the decoupling of the computation on clients and on the server, which is a key feature of federated learning.

To better handle non-I.I.D. data, one needs to introduce some other techniques. In non-I.I.D. scenarios,
the gradients have different distributions across clients. A natural idea is to bring in some extra parameters
which update along with the model parameters to make corrections (modifications) to the gradients on clients,
reducing their variance and further accelerating the convergence. This technique is the so-called **variance reduction**
technique [:footcite:ct:`johnson2013accelerating`], which was first introduced to federated learning in
[:footcite:ct:`karimireddy2020scaffold`] in the form of a new federated learning algorithm called **SCAFFOLD**
(Stochastic Controlled Averaging algorithm). The pseudocode for **SCAFFOLD** is shown as follows:

.. _pseduocode-scaffold:

.. image:: ../generated/algorithms/scaffold.svg
   :align: center
   :width: 80%
   :alt: Psuedocode for ``Scaffold``
   :class: no-scaled-link

Variance reduction is a technique that can be flexibly combined with most algorithms and has been widely used
in federated learning for dealing with statistical heterogeneity. However, it should be noted in the
`SCAFFOLD algorithm <pseduocode-scaffold_>`_ that on both the server and the clients, there are extra parameters
:math:`c` and :math:`c_k` to maintain, which may increase the communication cost. In scenarios which are sensitive
to communication cost, this would potentially be a problem. Therefore, a better solution could be a combination of
the variance reduction technique and some **skipping** techniques (e.g. [:footcite:ct:`proxskip-vr`]),
which will be introduced in next sections.

.. footbibliography::
