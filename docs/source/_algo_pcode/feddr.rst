.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{FedDR}}
    \begin{algorithmic}
    \REQUIRE step size $s = \frac{1}{\mu} > 0$, $\alpha \in (0, 2)$, error bounds $\varepsilon_{k,0} \geqslant 0$

    \STATE {\bfseries Initiation:}
    \STATE $\hspace{1.3em}$ {\bfseries Init server:} global model parameters $\theta^{(0)} \in \text{dom}(f)$, $\overline{\theta}^{(0)} = \widetilde{\theta}^{(0)} = \omega^{(0)} = \theta^{(0)} \in \mathbb{R}^d$
    \STATE $\hspace{1.3em}$ {\bfseries Init clients:} $\omega_k^{(0)} = \theta^{(0)}$, $\theta_k^{(0)} \approx \mathbf{prox}_{f_k, \mu}(\omega_k^{(0)})$, $\widehat{\theta}_k^{(0)} = 2\theta_k^{(0)} - \omega_k^{(0)}, ~ \forall k \in [K]$

    \FOR{$t = 0, 1, \cdots, T-1$}
        \STATE $\mathcal{S}^{(t)} \gets$ (random set of clients) $\subseteq [K]$
        \STATE each client $k \in \mathcal{S}^{(t)}$ receives $\overline{\theta}^{(t)}$ from server \COMMENT{communication}
        \FOR{each client $k \in \mathcal{S}^{(t)}$ {\bfseries in parallel}}
            \STATE choose $\varepsilon_{k,t+1} \geqslant 0$
            \STATE update $\omega_k^{(t+1)} \gets \omega_k^{(t)} + \alpha(\overline{\theta}^{(t)} - \theta_k^{(t)})$
            \STATE $\theta_k^{(t+1)} \approx \mathbf{prox}_{f_k, \mu}(y_k^{(t+1)})$ \COMMENT{inexact local prox step with error bound $\varepsilon_{k,0}$}
            \STATE $\widehat{\theta}_k^{(t+1)} \gets 2\theta_k^{(t+1)} - \omega_k^{(t+1)}$
            \STATE send $\Delta \widehat{\theta}_k^{(t)} = \widehat{\theta}_k^{(t+1)} - \widehat{\theta}_k^{(t)}$ to server
        \ENDFOR
        \STATE {\bfseries Server Update:}
        \STATE $\hspace{1.3em}$ $\omega^{(t+1)} \gets \omega^{(t)} + \alpha (\overline{\theta}^{(t)} - \omega^{(t)})$
        \STATE $\hspace{1.3em}$ $\widetilde{\theta}^{(t+1)} \gets \widetilde{\theta}^{(t)} + \frac{1}{K}\sum_{k \in \mathcal{S}^{(t)}} \Delta \widehat{\theta}_k^{(t)}$
        \STATE $\hspace{1.3em}$ $\overline{\theta}^{(t+1)} \gets \mathbf{prox}_{g, \frac{K+1}{Ks}} \left( \frac{K}{K+1} \widetilde{\theta}^{(t+1)} + \frac{1}{K+1} \omega^{(t+1)} \right)$
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
