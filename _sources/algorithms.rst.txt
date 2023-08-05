Federated learning algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview of Optimization Algorithms in Federated Learning
---------------------------------------------------------

Federated Optimization algorithms have been the central problem in the field of federated learning since its inception.
The most important contribution of the initial work on federated learning :cite:p:`mcmahan2017fed_avg` was the introduction of the Federated Averaging algorithm (FedAvg).

Mathematically, federated learning solves the following problem of minimization of empirical risk function

.. math::
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
   \begin{array}{cl}
   \minimize\limits_{\theta \in \R^d} & f(\theta) = \sum\limits_{k=1}^K w_k f_k(\theta).
   \end{array}

For further simplicity, we often take :math:`w_k = \frac{1}{K}`. The functions :math:`f_k` and :math:`f` are usually assumed to satisfy the following conditions:

   * (A1) :math:`f_k` and :math:`f` are :math:`L`-smooth (:math:`L > 0`), i.e. they have :math:`L`-Lipschitz continuous gradients:
   
      .. math::
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
         f(\theta) \geqslant c > -\infty, ~ \forall \theta \in \R^d,

      or equivalently,
      
         .. math::
            f^* := \inf\limits_{\theta \in \R^d} f(\theta) > - \infty.

In many cases, in order to facilitate the analysis of convergence, we will also make some assumptions about the gradient of the objective function:

   * (A3) Bounded gradient: there exists a constant :math:`G > 0` such that

      .. math::
         \lVert \nabla f_k (\theta) \rVert^2 \leqslant G^2, ~ \forall \theta \in \R^d, ~ k = 1, \ldots K.

And the following assumptions on data distributions:

   * (A4-1) Data distribution is I.I.D. (identically and independently distributed) across clients, i.e.
   
      .. math::
         \nabla f(\theta) = \expectation [f_k(\theta)] = \expectation\limits_{(x, y) \sim \mathcal{D}_k}[\nabla \ell_k(\theta; x, y)], ~ \forall \theta \in \R^d, ~ k = 1, \ldots K,

      or equivalently, for any :math:`\varepsilon > 0`, there exists a constant :math:`B \geqslant 0` such that

      .. math::
         \sum\limits_{k=1}^K \lVert \nabla f_k(\theta) \rVert^2 = \lVert f(\theta) \rVert^2, ~ \forall \theta \in \left\{ \theta \in \R^d ~ \middle| ~ \lVert f(\theta) \rVert^2 > \varepsilon \right\}.
   * (A4-2) Data distribution is non-I.I.D across clients, in which case we need a quantity to measure the degree of this statistical heterogeneity. This quantity can be defined in a number of ways :cite:`karimireddy2020scaffold, zhang2020fedpd, li2019convergence, sahu2018fedprox`. For example, in :cite:p:`karimireddy2020scaffold` and :cite:p:`zhang2020fedpd`, the so-called bounded gradient dissimilarity (BGD), denoted as :math:`(G; B)`-BGD, is used as this quantity. More specifically, there exists constants :math:`G > 0` and :math:`B \geqslant 0` such that

      .. math::
         \dfrac{1}{K} \sum\limits_{k=1}^K \lVert \nabla f_k(\theta) \rVert^2 \leqslant G^2 + B^2 \lVert \nabla f(\theta) \rVert^2, ~ \forall \theta \in \R^d.

      It should be noted that letting :math:`B = 0`, the bounded gradient dissimilarity condition (A4-2) degenrates to the bounded gradient condition (A3).

Sometimes, in the proof of algorithm convergence, one needs to make assumptions on the convexity of the objective function :math:`f`, which can be defined as follows:

   * (A5-1) convexity:

      .. math::
         f(a \theta + (1 - a) \theta') \leqslant a f(\theta) + (1 - a) f(\theta'), ~ \forall \theta, \theta' \in \mathcal{C}, ~ \alpha \in [0, 1].

      where :math:`\mathcal{C}` is a convex set on which :math:`f` is defined.
   * (A5-2) Strong convexity: there exists a constant :math:`\mu > 0` such that :math:`f - \frac{\mu}{2} \lVert \theta \rVert^2` is convex. In this case, we say that :math:`f` is :math:`\mu`-strongly convex.

Due to the natural layered and decoupled structure of the federal learning problem, it is more natural to consider the following constrained optimization problem:

.. math::
   \begin{array}{cl}
   \minimize & \frac{1}{K} \sum\limits_{k=1}^K f_k(\theta_k), \\
   \text{subject to} & \theta_k = \theta, ~ k = 1, \ldots, K.
   \end{array}

Federated Averaging Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

to write....

.. image:: ./generated/algorithms/fedavg.svg
   :align: center
   :width: 80%
   :alt: Psuedocode for FedAvg

.. image:: ./generated/algorithms/fedopt.svg
   :align: center
   :width: 80%
   :alt: Psuedocode for FedOpt

Proximal Algorithms in Federated Learning
-----------------------------------------

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

to write....

Primal-Dual Algorithms in Federated Learning
--------------------------------------------

to write....

Operator Splitting Algorithms in Federated Learning
---------------------------------------------------

to write....

Skipping Algorithms in Federated Learning
---------------------------------------------------

to write....

.. bibliography::
