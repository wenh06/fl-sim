.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{SCAFFOLD} (Stochastic Controlled Averaging)}
    \begin{algorithmic}
    \REQUIRE server learning rate $\eta_g,$ client learning rates $\eta_k,$ $\forall k \in [K]$

    \STATE {\bfseries Initiation:}
    \STATE $\hspace{1.3em}$ {\bfseries Server init:} global model parameters $\theta^{(0)} \in \mathbb{R}^d,$ control variates $c^{(0)} \in \mathbb{R}^d$
    \STATE $\hspace{1.3em}$ {\bfseries Clients init:} control variates $c_k^{(0)} \in \mathbb{R}^d, ~ \forall k \in [K]$

    \FOR{each round $t = 0, 1, \cdots, T-1$}
        \STATE $\mathcal{S}^{(t)} \gets$ (random set of clients) $\subseteq [K]$
        \STATE broadcast $(\theta^{(t)}, c^{(t)})$ to clients $k \in \mathcal{S}^{(t)}$
        \FOR{each client $k \in \mathcal{S}^{(t)}$ {\bfseries in parallel}}
            \STATE $\theta_k^{(t, 0)} \gets \theta^{(t)}$
            \FOR{local step $r = 0, 1, \cdots, R-1$}
                \COMMENT{$R$ iterates of SGD with variance reduction}
                \STATE compute mini-batch gradient $g_k^{(t, r)}$ of $\nabla f_k(\theta_k^{(t, r)})$
                \STATE $\theta_k^{(t, r+1)} \gets \theta_k^{(t, r)} - \eta_k \left( g_k^{(t, r)} - c_k^{(t)} + c^{(t)} \right)$
            \ENDFOR
            \STATE $c_k^{(t+\frac{1}{2})} \gets \begin{cases} \text{Option 1} & g_k^{(t, 0)} \\ \text{Option 2} & c_k^{(t)} - c^{(t)} + \frac{1}{R\eta_k} \left( \theta^{(t)} - \theta_k^{(t, R)} \right) \end{cases}$
            \STATE $( \Delta \theta_k^{(t)}, \Delta c_k^{(t)} ) \gets ( \theta_k^{(t, R)} - \theta^{(t)}, c_k^{(t+\frac{1}{2})} - c_k^{(t)} )$
            \STATE send $( \Delta \theta_k^{(t)}, \Delta c_k^{(t)} )$ to server
            \STATE $c_k^{(t+1)} \gets c_k^{(t+\frac{1}{2})}$
        \ENDFOR
        \STATE {\bfseries Server Update:}
        \STATE $\hspace{1.3em}$ $( \Delta \theta^{(t)}, \Delta c^{(t)} ) \gets \frac{1}{\lvert \mathcal{S}^{(t)} \rvert} \sum\limits_{k \in \mathcal{S}^{(t)}} ( \Delta \theta_k^{(t)}, \Delta c_k^{(t)} )$
        \STATE $\hspace{1.3em}$ $\theta^{(t+1)} \gets \theta^{(t)} + \eta_g \Delta \theta^{(t)}$
        \STATE $\hspace{1.3em}$ $c^{(t+1)} \gets c^{(t)} + \frac{\lvert \mathcal{S}^{(t)} \rvert}{K} \Delta c^{(t)}$
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
