Federated learning algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview of Optimization Algorithms in Federated Learning
---------------------------------------------------------

to write....

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
