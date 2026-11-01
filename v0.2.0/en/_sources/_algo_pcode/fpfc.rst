.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{FPFC}}
    \begin{algorithmic}
    \REQUIRE number of devices $m,$ number of rounds $K,$ local epochs $T_i,$ stepsize $\alpha,$ penalty parameters $\xi, \lambda, a,$ and $\rho$

    \STATE {Initiation:}
    \STATE $\omega_1^0 = \cdots = \omega_m^0$ on the clients
    \STATE $\zeta_i^0 = \omega_i^0$ for $i \in [m]$, $\theta_{ij}^0 = 0$ and $v_{ij}^0 = 0$ for $i, j \in [m]$ on the server

    \FOR{each round $k = 0, 1, \cdots, K-1$}
    \STATE the server randomly chooses a subset of devices $\mathcal{A}_k$
    \STATE the server downloads $\zeta_i^k$ to each device $i \in \mathcal{A}_k$
    \STATE \FOR{each device $i \in \mathcal{A}_k$ {in parallel}}
    \STATE $\omega_i^{k,0} = \omega_i^k$
    \STATE \FOR{$t = 0, 1, \cdots, T_i - 1$}
    \STATE $\hspace{1.3em}$ $\omega_i^{k,t+1} = \omega_i^{k,t} - \alpha \left[ \nabla f_i(\omega_i^{k,t}) + \rho \left( \omega_i^{k,t} - \zeta_i^k \right) \right]$
    \STATE \ENDFOR
    \STATE $\omega_i^{k+1} = \omega_i^{k,T_i}$
    \STATE \ENDFOR
    \STATE each device $i \in \mathcal{A}_k$ uploads $\omega_i^{k+1}$ to the server
    \STATE {Server Update:}
    \STATE \FOR{$i, j \in \mathcal{A}_k$ ($i < j$)}
    \STATE $\hspace{1.3em}$ $\delta_{ij}^{k+1} = \omega_i^{k+1} - \omega_j^{k+1} + v_{ij}^k / \rho$
    \STATE $\hspace{1.3em}$ $\theta_{ij}^{k+1} \gets \underset{\theta}{\text{prox}}$ of the (scaled) fusion penalty at $\delta_{ij}^{k+1}$ (Eq. (6) of the paper)
    \STATE $\hspace{1.3em}$ $v_{ij}^{k+1} = v_{ij}^k + \rho \left( \omega_i^{k+1} - \omega_j^{k+1} - \theta_{ij}^{k+1} \right)$
    \STATE $\hspace{1.3em}$ $\theta_{ji}^{k+1} = -\theta_{ij}^{k+1}, ~ v_{ji}^{k+1} = -v_{ij}^{k+1}$
    \STATE \ENDFOR
    \STATE for $i \notin \mathcal{A}_k$ or $j \notin \mathcal{A}_k$: $\theta_{ij}^{k+1} = \theta_{ij}^k, ~ v_{ij}^{k+1} = v_{ij}^k$
    \STATE for $i \in [m]$: $\hspace{0.5em}$ $\zeta_i^{k+1} = \frac{1}{m} \sum\limits_{j=1}^m \left( \omega_j^{k+1} + \theta_{ij}^{k+1} - v_{ij}^{k+1} / \rho \right)$
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
