.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{FedDyn}}
    \begin{algorithmic}
    \REQUIRE penalty coeffecient $\mu$

    \STATE {\bfseries Initiation:}
    \STATE $\hspace{1.3em}$ {\bfseries Init server:} global model parameters $\theta^{(0)} \in \mathbb{R}^d,$ $h = 0 \in \mathbb{R}^d$
    \STATE $\hspace{1.3em}$ {\bfseries Init clients:} local gradient $\mathfrak{g}_k^{(0)} \gets 0 \in \mathbb{R}^d, ~ \forall k \in [K]$

    \FOR{each round $t = 0, 1, \cdots, T-1$}
        \STATE $\mathcal{S}^{(t)} \gets$ (random set of clients) $\subseteq [K]$
        \STATE broadcast $\theta^{(t)}$ to clients $k \in \mathcal{S}^{(t)}$
        \FOR{each client $k \in \mathcal{S}^{(t)}$ {\bfseries in parallel}}
            \STATE $\theta_k^{(t+1)} \gets \underset{\theta_k}{\text{argmin}} \left\{ f_k(\theta_k) - \langle \mathfrak{g}_k^{(t)}, \theta_k \rangle + \frac{\mu}{2} \lVert \theta_k - \theta^{(t)} \rVert^2 \right\}$
            \STATE $\mathfrak{g}_k^{(t+1)} \gets \mathfrak{g}_k^{(t)} - \mu (\theta_k^{(t+1)} - \theta^{(t)})$ \COMMENT{update local gradient}
            \STATE send $\theta_k^{(t+1)}$ to server
        \ENDFOR
        \STATE {\bfseries Server Update:}
        \STATE $\hspace{1.3em}$ $h^{(t+1)} \gets h^{(t)} - \frac{\mu}{K} \left(\sum\limits_{k \in \mathcal{S}^{(t)}} \theta_k^{(t+1)} - \theta^{(t)} \right)$
        \STATE $\hspace{1.3em}$ $\theta^{(t+1)} \gets \left( \frac{1}{\# \mathcal{S}^{(t)}}\sum\limits_{k \in \mathcal{S}^{(t)}} \theta_k^{(t+1)} \right) - \frac{1}{\mu} h^{(t+1)}$
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
