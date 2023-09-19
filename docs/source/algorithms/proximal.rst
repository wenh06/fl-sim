.. _fl_alg_proximal:

Proximal Algorithms in Federated Learning
-----------------------------------------

In non-I.I.D. scenarios, based on the idea of reducing the impact of local updates of clients on the global model,
[:footcite:ct:`sahu2018fedprox`] first introduced a proximal term to the local objective functions, aiming at making the
algorithm more stable and converging faster. Compared to ``SCAFFOLD``, methods using proximal terms do not need to
maintain extra parameters (mainly related to the gradients), hence having no communication overhead and no
additional cost to security (refer to [:footcite:ct:`zhu2019deep_leakage`] for more details).

To be more specific, in the :math:`(t+1)`-th iteration, the local objective function of client :math:`k` changes from
:math:`f_k(\theta_k)` to the following form with a proximal term:

.. math::
   :label: fedprox-local-obj

   \DeclareMathOperator*{\expectation}{\mathbb{E}}
   \DeclareMathOperator*{\minimize}{minimize}
   \newcommand{\R}{\mathbb{R}}

   h_k(\theta_k, \theta^{(t)}) := f_k(\theta_k) + \frac{\mu}{2} \lVert \theta_k - \theta^{(t)} \rVert^2,

where :math:`\mu` is a penalty constant. It should be noticed that the proximal center :math:`\theta^{(t)}` is
the model parameter on the server node obtained in the previous iteration (the :math:`t`-th iteration). Indeed,
the overall optimization problem can be modeled as the following constrained optimization problem

.. math::
   :label: fedprox-whole

   \begin{array}{cl}
   \minimize & \frac{1}{K} \sum\limits_{k=1}^K \left\{ f_k(\theta_k) + \frac{\mu}{2} \lVert \theta_k - \theta \rVert^2 \right\} \\
   \text{subject to} & \theta = \frac{1}{K} \sum\limits_{k=1}^K \theta_k.
   \end{array}

For alternatives for the proximal center, studies were conducted in [:footcite:ct:`hanzely2020federated, li_2021_ditto`] which would be
introduced later. Now, we summarize the pseudocode for ``FedProx`` as follows:

.. _pseduocode-fedprox:

.. image:: ../generated/algorithms/fedprox.svg
   :align: center
   :width: 80%
   :alt: Psuedocode for ``FedProx``
   :class: no-scaled-link

We denote the :math:`\gamma`-inexact solution :math:`\theta_k^{(t)}` as

.. math::
   :label: prox-op

   \DeclareMathOperator*{\argmax}{arg\,max}
   \DeclareMathOperator*{\argmin}{arg\,min}
   % \DeclareMathOperator*{\prox}{prox}
   \newcommand{\prox}{\mathbf{prox}}

   \theta_k^{(t)} \approx \prox_{f_k, \mu} (\theta^{(t)}) := \argmin\limits_{\theta_k} \left\{ f_k(\theta_k) + \frac{\mu}{2} \lVert \theta_k - \theta^{(t)} \rVert^2 \right\},

where :math:`\prox_{f_k, \mu}` is the proximal operator [:footcite:ct:`Moreau_1965_prox`] of :math:`f_k` with respect to :math:`\mu`.
Let :math:`s = \frac{1}{\mu}`, since one has :math:`\prox_{f_k, \mu} = \prox_{sf_k, 1}`, we also denote :math:`\prox_{f_k, \mu}`
as :math:`\prox_{sf_k}`. Corresponding function

.. math::
   :label: moreau_env

   \mathcal{M}_{sf_k} (\theta^{(t)}) = \mathcal{M}_{f_k, \mu} (\theta^{(t)}) := \inf\limits_{\theta_k} \left\{ f_k(\theta_k) + \frac{\mu}{2} \lVert \theta_k - \theta^{(t)} \rVert^2 \right\}

is called **Moreau envelope** or **Moreau-Yosida regularization** of :math:`f_k` with respect to :math:`\mu`.
Moreau envelope of a function :math:`f_k` has the following relationship [:footcite:ct:`Parikh_2014_pa`] with its proximal operator:

.. math::
   :label: prox-moreau-relation
   
   \prox_{sf_k} (\theta) = \theta - s \nabla \mathcal{M}_{sf_k} (\theta), ~ \forall \theta \in \R^d.

Namely, :math:`\prox_{sf_k}` can be regarded as the gradient descent operator for minimizing :math:`\mathcal{M}_{sf_k}` with step size :math:`s`.

For the convergence of ``FedProx`` in non-I.I.D. scenarios, [:footcite:ct:`sahu2018fedprox`] has the following theorem:

.. _fedprox_thm4:

.. proof:theorem:: [:footcite:ct:`sahu2018fedprox`] Theorem 4

   Assume that the objective functions on clients :math:`\{f_k\}_{k=1}^K` are non-convex, :math:`L`-smooth (definition see :eq:`l-smooth`), and there exists a constant :math:`L_- > 0` such that :math:`\nabla^2 f_k \succcurlyeq -L_- I_d`.
   Assume further that the functions :math:`\{f_k\}_{k=1}^K` satisfy the so-called bounded dissimilarity condition, i.e.
   for any :math:`\varepsilon > 0`, there exists a constant :math:`B_{\varepsilon} > 0` such that for any point :math:`\theta`
   in the set :math:`\mathcal{S}_{\varepsilon}^c := \{ \theta ~|~ \lVert \nabla f(\theta) \rVert^2 > \varepsilon\}`,
   the following inequality holds

   .. math::
      :label: fedprox_bdd_dissim

      B(\theta) := \frac{\expectation_k [\lVert \nabla f_k(\theta) \rVert^2]}{\lVert \nabla f(\theta) \rVert^2} \leqslant B_{\varepsilon}.

   Fix constants :math:`\mu, \gamma` satisfying

   .. math::
      :label: fedprox_mu_gamma

      \rho := \left( \frac{1}{\mu} - \frac{\gamma B}{\mu} - \frac{B(1+\gamma)\sqrt{2}}{\bar{\mu}\sqrt{K}} - \frac{LB(1+\gamma)}{\bar{\mu}\mu} - \frac{LB^2(1+\gamma)^2}{2\bar{\mu}^2} - \frac{LB^2(1+\gamma)^2}{\bar{\mu}^2 K} \left( 2\sqrt{2K} + 2 \right) \right) > 0,

   where :math:`\bar{\mu} = \mu - L_- > 0`. Then, in the :math:`(t+1)`-th iteration of ``FedProx``, assuming that the global model
   :math:`\theta^{(t)}` of the previous iteration is not the first-order stationary point of the global objective function :math:`f(\theta)`,
   (i.e. :math:`\theta^{(t)} \in \mathcal{S}_{\varepsilon}^c`), the following decrease in the global objective function holds

   .. math::
      :label: fedprox_obj_decrease

      \expectation\nolimits_{\mathcal{S}^{(t)}}[f(\theta^{(t+1)})] \leqslant f(\theta^{(t)}) - \rho \lVert \nabla f (\theta^{(t)}) \rVert^2.

.. _fedprox_rem1:

.. proof:remark::

   For the `convergence theorem <fedprox_thm4_>`_ of ``FedProx``, we have the following observations: in a neighbourhood of
   some zero of :math:`\lVert \nabla f \rVert`, if this zero is not cancelled by :math:`\mathbb{E}_k[\lVert \nabla f_k \rVert]`,
   i.e. this point is also a zero of :math:`\mathbb{E}_k[\lVert \nabla f_k \rVert]` with the same or higher multiplicity,
   then in the neighbourhood, :math:`B_{\varepsilon}` goes rapidly to infinity as :math:`\varepsilon \to 0`, thus violating
   the condition :math:`\rho > 0`. In this case, the inequality :eq:`fedprox_obj_decrease` becomes meaningless.

   When the data distribution across clients is identical (ideal case), then :math:`B_{\varepsilon}` is constantly equal to 1,
   which would not have the problem mentioned above. This problem is the start point of a series of follow-up works [:footcite:ct:`pathak2020fedsplit,tran2021feddr`].

The positive significance of the ``FedProx`` algorithm is that it first introduced the proximal point algorithms (PPA) in the field of
federated learning, although which were only used for solving local optimization problems (or equivalently the inner loop problem) and the
whole of the ``FedProx`` algorithm is not a PPA in strict sense. The ``FedProx`` algorithm provides not only a good framework for theoretical
analysis, but also a good starting point for the design of new algorithms. A large proportion of the algorithms proposed later for personalized
fedrated learning [:footcite:ct:`hanzely2020federated, acar2021feddyn, li_2021_ditto, t2020pfedme, li2021pfedmac`] rely on the proximal terms (or similar terms)
as the main technical tool for personalization.

.. _fig-apfl:

.. tikz:: Schematic diagram for :math:`f_k(\alpha_k \omega_k + (1 - \alpha_k) \theta^*)` in the APFL algorithm.
   :align: center
   :xscale: 80
   :libs: arrows.meta,positioning,calc

   \tiny
   % \coordinate (origin) at (0, 0);
   \coordinate (rect1) at (-5, -3);
   \coordinate (rect2) at (5, 3);
   \fill [gray!20] (-1.7, -2.5) rectangle (4, 3);
   \fill [gray!50] (-0.4, -1.5) rectangle (2.45, 1.25);
   \draw (rect1) rectangle (rect2);
   \node at (3.2, 2.5) {$\mathrm{dom} f_k$};
   \node at (-4, -2.5) {$\widetilde{\mathrm{dom} f_k}$};
   \node at (-4, 2.5) {$\alpha = \frac{1}{2}$};
   \draw[] plot [smooth cycle] coordinates {(-1, 0) (-0.7, -0.1) (-0.3, 0.2) (-0.5, 0.3) (-1.1, 0.1)};
   \draw[] plot [smooth cycle] coordinates {(-1.3, -0.3) (-0.6, -0.6) (0.3, 0.7) (-0.1, 0.9) (-0.7, 0.7) (-1.6, 0.2)};
   \draw[] plot [smooth cycle] coordinates {(-1.6, -0.5) (-0.2, -1.2) (0.9, 1.2) (0.6, 1.6) (-0.7, 1.2) (-2.2, 0.3)};
   \draw[] plot [smooth cycle] coordinates {(-2.4, -1.1) (0.6, -2.6) (1.9, 1.9) (1.1, 2.7) (-0.9, 1.9) (-3.1, 0.1)};
   \node at (0.9, -0.5) (theta) [circle, fill=black, inner sep=0pt, minimum size=5pt, label=below:{$\theta^*$}] {};
   \node at (-2.1, 0.6) (omega1) [circle, fill=black, inner sep=0pt, minimum size=5pt, label=left:{$\omega_k$}] {};
   \draw[dashed, thin] (theta) -- (omega1);
   \draw plot[only marks, mark=triangle*, mark size=4pt, thick] coordinates {(-0.7, 0.1)};
   \begin{scope}
   \clip (rect1) rectangle (rect2);
   \draw[dashed, thin] (theta) circle (0.5);
   \draw[dashed, thin] (theta) circle (1.1);
   \draw[dashed, thin] (theta) circle (1.9);
   \draw[dashed, thin] (theta) circle (3.2);
   \end{scope}

.. _fig-feddyn:

.. tikz:: Client model parameter update schematic diagram of the FedDyn algorithm.
   :align: center
   :xscale: 80
   :libs: arrows.meta,positioning,calc

   % \fontsize{1.5}{2.5}\selectfont
   \tiny
   % \coordinate (origin) at (0, 0);
   \coordinate (rect1) at (-4, -2.1);
   \coordinate (rect2) at (3.6, 2.3);
   \draw (rect1) rectangle (rect2);
   \node at (-3.2, -1.9) {$\mathrm{dom} f_k$};
   \begin{scope}
   \clip (rect1) rectangle (rect2);
   \draw[] plot [smooth cycle] coordinates {(-1, 0) (-0.7, -0.1) (-0.3, 0.2) (-0.5, 0.3) (-1.1, 0.1)};
   % \draw[] plot [smooth cycle] coordinates {(-1.3, -0.3) (-0.6, -0.6) (0.3, 0.7) (-0.1, 0.9) (-0.7, 0.7) (-1.6, 0.2)};
   \draw[] plot [smooth cycle] coordinates {(-1.6, -0.5) (-0.2, -1.2) (0.9, 1.2) (0.6, 1.6) (-0.7, 1.2) (-2.2, 0.3)};
   \draw[] plot [smooth cycle] coordinates {(-2.4, -1.1) (0.6, -2.6) (1.9, 1.9) (1.1, 2.7) (-0.9, 1.9) (-3.1, 0.1)};
   \end{scope}
   \node at (0.9, -0.5) (global) [circle, fill=black, inner sep=0pt ,minimum size=5pt, label=below:{$\theta^{(t)}$}] {};
   \node at (1.2, 0.5) (local) [circle, fill=black, inner sep=0pt, minimum size=5pt, label=above:{$\theta_k^{(t)}$}] {};
   \coordinate (min1) at (-0.7, 0.1);
   \draw plot[only marks, mark=triangle*, mark size=4pt, thick] coordinates {(min1)};
   \coordinate (min2) at (0.1, -0.6);
   \draw plot[only marks, mark=star, mark size=4pt, thick] coordinates {(min2)};
   \path (local) edge [draw, dashed, -{Stealth}] ($(local)!0.6!(min1)$);
   \node at ($(local)!0.6!(min1)$) (grad) [label=above:{$\mathrm{g}_k^{(t)}$}] {};
   \path (local) edge [draw, dashed, -{Stealth}] ($(local)!0.7!(min2)$);
   % \node at (0.3, 0.0) (next) [circle, fill=black, inner sep=0pt, minimum size=5pt, label=left:{$\theta_k^{(t+1)}$}] {};
   \node at (0.3, 0.0) (next) [circle, fill=black, inner sep=0pt, minimum size=5pt] {};
   \node at (0.1, -0.25) {$\theta_k^{(t+1)}$};
   \path (local) edge [draw, thick, -{Stealth}, decorate, decoration={snake, amplitude=1.5pt, pre length=4pt, post length=3pt}] (next);
   \begin{scope}
   \clip (rect1) rectangle (rect2);
   \draw[dashed, thin] (global) circle (0.7);
   % \draw[dashed, thin] (global) circle (1.1);
   \draw[dashed, thin] (global) circle (1.9);
   \draw[dashed, thin] (global) circle (3.2);
   \end{scope}

to write more....

.. footbibliography::
