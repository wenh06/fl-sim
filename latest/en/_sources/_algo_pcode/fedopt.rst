.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{FedOpt}}
    \begin{algorithmic}
    \REQUIRE methods {\bfseries ServerOpt, ClientOpt}, learning rates (schedule) $\eta_g, \eta_l$

    \STATE {\bfseries Initiation:} global (server) model parameters $\theta^{(0)} \in \mathbb{R}^d$

    \FOR{each round $t = 0, 1, \cdots, T-1$}
        \STATE $\mathcal{S}^{(t)} \gets$ (random set of clients) $\subseteq [K]$
        \STATE broadcast $\theta^{(t)}$ to clients $k \in \mathcal{S}^{(t)}$
        \FOR{each client $k \in \mathcal{S}^{(t)}$ {\bfseries in parallel}}
            \STATE $\theta_k^{(t, 0)} \gets \theta^{(t)}$
            \FOR{local step $r = 0, 1, \cdots, R-1$}
                \STATE compute unbiased estimate $g_k^{(t, r)}$ of $\nabla f_k(\theta_k^{(t, r)})$
                \STATE $\theta_k^{(t, r+1)} \gets$ {\bfseries ClientOpt}$(\theta_k^{(t, r)}, g_k^{(t, r)}, \eta_l, t)$
            \ENDFOR
            \STATE $\Delta_{k}^{(t)} \gets \theta_k^{(t, R)} - \theta^{(t)}$
            \STATE send $\Delta_{k}^{(t)}$ to server
        \ENDFOR
        \STATE {\bfseries Server Update:}
        \STATE $\hspace{1.3em}$ $\Delta^{(t)} \gets \text{aggregate} \left( \left\{ \Delta_{k}^{(t)} \right\}_{k \in \mathcal{S}^{(t)}} \right)$
        \STATE $\hspace{1.3em}$ $\theta^{(t+1)} \gets$ {\bfseries ServerOpt}$(\theta^{(t)}, \Delta^{(t)}, \eta_g, t)$
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
