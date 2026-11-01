.. _fl_alg_notation:

Notation
--------

This page summarizes the notations used throughout the theory pages of this chapter; the same
notation is used in the accompanying research report. Symbols not listed here are explained
where they first appear.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Notation
     - Meaning
   * - :math:`\N`
     - the set of natural numbers (including :math:`0`)
   * - :math:`\R`
     - the set of real numbers
   * - :math:`\R^d`
     - the :math:`d`-dimensional real vector space (of column vectors)
   * - :math:`\# S`
     - the cardinality of a finite set :math:`S`
   * - :math:`\Pi_{\Omega}`
     - the projection map onto a convex set :math:`\Omega \subseteq \R^d`
   * - :math:`[K]`
     - the set :math:`\{1, 2, \ldots, K\}`, where :math:`K \in \N_{> 0}`
   * - :math:`\theta`
     - the model parameters, :math:`\theta \in \R^d`
   * - :math:`\col(\theta_1, \ldots, \theta_K)`
     - the column vector stacking :math:`\theta_1, \ldots, \theta_K` vertically in order
   * - :math:`\operatorname{dom}(f)`
     - the domain of the function :math:`f`, i.e. :math:`\{\theta \in \R^d ~|~ f(\theta) < +\infty\}`
   * - :math:`\nabla f(\cdot)`
     - the gradient of :math:`f`
   * - :math:`\partial f(\cdot)`
     - the subdifferential (the set of subgradients) of :math:`f`
   * - :math:`\mathcal{D}(x, y)`
     - the (joint) distribution of the data, where :math:`x` is the feature and :math:`y` is the label
   * - :math:`I_n` or :math:`I`
     - the identity matrix (of order :math:`n`)
   * - :math:`O_n` or :math:`O`
     - the zero matrix (of order :math:`n`)
   * - :math:`A^{\mathrm{T}}`
     - the transpose of a matrix :math:`A`
   * - :math:`A^{-1}`
     - the inverse of a matrix :math:`A`
   * - :math:`\lVert \cdot \rVert_p`
     - the :math:`p`-norm of a vector or a matrix (:math:`0 \leqslant p \leqslant +\infty`);
       when the subscript :math:`p` is omitted, the :math:`2`-norm is meant
   * - :math:`\lVert \cdot \rVert_F`
     - the Frobenius norm of a matrix
   * - :math:`\mathbb{E}[\cdot]`
     - the mathematical expectation
   * - :math:`\mathcal{N}(\mu, \sigma^2)`
     - the univariate normal distribution with mean :math:`\mu \in \R` and variance :math:`\sigma^2 \in \R`
   * - :math:`\mathcal{N}(v, \Sigma)`
     - the :math:`d`-dimensional normal distribution with mean :math:`v \in \R^d` and
       covariance matrix :math:`\Sigma \in \R^{d \times d}`
