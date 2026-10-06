.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{ProxSkip}}
    \begin{algorithmic}
    \REQUIRE learning rate $\gamma,$ probability $p \in [0, 1)$

    \STATE {\bfseries Initiation:}
    \STATE $\hspace{1.3em}$ {\bfseries Server init:} global model parameters $\theta^{(0)} \in \mathbb{R}^d$
    \STATE $\hspace{1.3em}$ {\bfseries Clients init:} local model parameters $\theta_k^{(0)} \in \mathbb{R}^d,$ control variates $c_k^{(0)} \in \mathbb{R}^d, ~ \forall k \in [K]$

    \FOR{each round $t = 0, 1, \cdots, T-1$}
        \FOR{each client $k \in [K]$ {\bfseries in parallel}}
            \STATE $\theta_k^{(t, 0)} \gets \theta_k^{(t)}$
            \FOR{local step $r = 0, 1, \cdots, R-1$}
                \COMMENT{$R$ iterates of SGD with variance reduction}
                \STATE compute mini-batch gradient $g_k^{(t, r)}$ of $\nabla f_k(\theta_k^{(t, r)})$
                \STATE $\theta_k^{(t, r+1)} \gets \theta_k^{(t, r)} - \gamma \left( g_k^{(t, r)} - c_k^{(t)} \right)$
            \ENDFOR
            \STATE $\theta_k^{(t+\frac{1}{2})} \gets \theta_k^{(t, R)}$
        \ENDFOR
        \STATE with probability $1 − p$ do global communication
        \STATE $\hspace{1.3em}$ client $k$ send $\theta_{k}^{(t+\frac{1}{2})}$ to server $\forall k \in [K]$
        \STATE $\hspace{1.3em}$ {\bfseries Server Update:} $\theta^{(t+1)} \gets \frac{1}{K} \sum\limits_{k=1}^K \theta_{k}^{(t+\frac{1}{2})}$ \COMMENT{compute global average}
        \STATE $\hspace{1.3em}$ server broadcast $\theta^{(t+1)}$ to clients $k \in [K]$
        \STATE $\hspace{1.3em}$ on client $k:$ $\theta^{(t+1)}_{k} \gets \theta^{(t+1)}, ~ c_k^{(t+1)} \gets c_k^{(t)} + \frac{p}{\gamma}(\theta^{(t+1)}_{k} - \theta^{(t+\frac{1}{2})}_{k}), ~ \forall k \in [K]$
        \STATE with probability $p$ skip global communication:
        \STATE $\hspace{1.3em}$ {\bfseries Client Update:} $\theta^{(t+1)}_{k} \gets \theta_{k}^{(t+\frac{1}{2})}, ~ c_k^{(t+1)} \gets c_k^{(t)}, ~ \forall k \in [K]$
        \STATE $\hspace{1.3em}$ \COMMENT{on server, $\theta^{(t+1)} \gets \theta^{(t)}$}
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
