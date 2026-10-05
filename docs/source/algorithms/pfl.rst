.. _fl_alg_pfl:

Personalized Federated Learning Algorithms
---------------------------------------------------

In personalized federated learning (PFL), each client seeks a model tailored to its own data
distribution while still benefiting from the knowledge of the other clients. A simple and general
way to formulate PFL is via *weak consensus*: instead of enforcing full consensus
:math:`\theta_1 = \cdots = \theta_K`, one adds a regularizer :math:`\mathcal{R}` that penalizes the
deviation between the local models. This page collects the PFL algorithms (implemented in
``fl_sim`` or closely related) together with their pseudocodes.

.. _fl_alg_pfl_l2sgd:

``L2SGD``: Loopless Local SGD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. _pcode-pfedme:

.. include:: ../_algo_pcode/pfedme.rst

.. _pcode-ditto:

.. include:: ../_algo_pcode/ditto.rst

.. _pcode-apfl:

.. include:: ../_algo_pcode/apfl.rst

.. _pcode-feddyn:

.. include:: ../_algo_pcode/feddyn.rst

.. _pcode-pfedmac:

.. include:: ../_algo_pcode/pfedmac.rst

.. footbibliography::
