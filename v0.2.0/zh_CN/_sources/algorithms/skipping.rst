.. _fl_alg_skipping:

Skipping Algorithms in Federated Learning
---------------------------------------------------

Even the earliest federated learning algorithm, ``FedAvg``
[:footcite:ct:`mcmahan2017fed_avg`], performs multiple gradient-descent steps in the local inner
loop of its clients -- which, in a sense, is already a *skipping* update: the communication
round is skipped with probability :math:`1 - p = 0`. The ``ProxSkip`` algorithm
[:footcite:ct:`proxskip,proxskip-vr`] (named ``Scaffnew`` in the original paper) studies the
skipping technique systematically and makes the communication round skipped with a *nonzero*
probability. It is instructive to read ``ProxSkip`` as ``SCAFFOLD`` combined with the skipping
technique.

.. _fl_alg_skipping_proxskip:

``ProxSkip``/``Scaffnew``
~~~~~~~~~~~~~~~~~~~~~~~~~

From the optimization-theoretic viewpoint, model the federated learning problem as the consensus
optimization problem over the consensus set
:math:`\mathcal{E} = \{ \Theta \mid \theta_1 = \cdots = \theta_K \}`,

.. math::
   :label: skipping-consensus

   \DeclareMathOperator*{\minimize}{minimize}
   \newcommand{\col}{\operatorname{col}}
   \newcommand{\R}{\mathbb{R}}
   \newcommand{\prox}{\mathbf{prox}}

   \begin{array}{cl}
   \minimize & F(\Theta) = \frac{1}{K} \sum\limits_{k=1}^K f_k(\theta_k), \\
   \text{subject to} & \Theta \in \mathcal{E}, \\
   \text{where} & \Theta = \col(\theta_1, \cdots, \theta_K) \in \R^{Kd}.
   \end{array}

As in the :ref:`operator splitting section <fl_alg_operator_splitting>`, one can convert this
constrained problem into an unconstrained one with the indicator function
:math:`\iota_{\mathcal{E}}` of the consensus set and split the first-order optimality condition
:math:`0 \in \nabla F(\Theta) + \partial \iota_{\mathcal{E}}(\Theta)`. More generally, replacing
:math:`\iota_{\mathcal{E}}` by any CCP function :math:`g`, the fixed-point solver is the
*proximal gradient descent* (ProxGD)

.. math::
   :label: skipping-proxgd

   \Theta^{(t+1)} = \prox_{s g}\left( \Theta^{(t)} - s \nabla F(\Theta^{(t)}) \right),

for the problem :math:`\minimize \ F(\Theta) + g(\Theta)`. The practical difficulty of ProxGD is
that the proximal operator :math:`\prox_{s g}` of the CCP function :math:`g` is typically
expensive to evaluate [:footcite:ct:`friedlander2017efficient`]. In federated learning, this
proximal step *is* the communication: the model aggregation from the clients to the server
followed by the broadcast back. Skipping this performance-bottleneck step with some probability
is therefore a natural remedy, and this is exactly what ``ProxSkip`` does.

A remark on the control variates: unlike ``SCAFFOLD``, ``ProxSkip`` does not use (explicit)
gradient variance reduction. The primary role of the control variates :math:`c_k` is to
guarantee the *correct convergence* of the overall iteration: one can show
:math:`c_k \to \nabla f_k(\theta^*)` at a solution :math:`\theta^*` of the original problem, so
the fixed point of the iteration is a solution of the problem -- this is the key property.
``ProxSkip`` maintains no global control variate communicated between the server and the
clients; instead, whenever a communication happens, the local control variate
:math:`c_k` is *implicitly* updated by
:math:`\frac{p}{\gamma} \left( \theta^{(t+1)} - \theta_k^{(t + \frac{1}{2})} \right)`, which acts
as a global control variate. This is reminiscent of the Option II update rule of the local
control variates in ``SCAFFOLD``: it saves the extra parameter exchange between the server and
the clients, at the cost of the flexibility of the control variate updates. The pseudocode is
summarized below.

.. _pcode-proxskip:

.. include:: ../_algo_pcode/proxskip.rst

.. footbibliography::
