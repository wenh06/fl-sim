.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{FedPD}}
    \begin{algorithmic}
    \REQUIRE step size $s = \frac{1}{\mu} > 0,$ skip probability $p \in [0, 1)$

    \STATE {\bfseries Initiation:}
    \STATE $\hspace{1.3em}$ {\bfseries Init server:} global parameters $\theta^{(0)} \in \mathbb{R}^d$
    \STATE $\hspace{1.3em}$ {\bfseries Init clients:} local parameters$\theta_{0,k}^{(0)} \in \mathbb{R}^d,$ dual variables $\lambda_k^{(0)} \in \mathbb{R}^d,$ $\forall k \in [K]$

    \FOR{each round $t = 0, 1, \cdots$}
        \FOR{each client $k = 1, \cdots, K$ {\bfseries in parallel}}
            \STATE $\theta_k^{(t+1)} \gets \textbf{Oracle}_k(\mathcal{L}_k(\theta_{k, 0}^{(t)}, \theta_k, \lambda_k^{(t)}), \theta_k^{(t)})$ \COMMENT{$\textbf{Oracle}_k$ can be SGD, etc.}
            \STATE $\lambda_k^{(t+1)} \gets \lambda_k^{(t)} + \frac{1}{s} (\theta_k^{(t+1)} - \theta_{k, 0}^{(t)})$ \COMMENT{dual update step}
            \STATE $\theta_{k, 0}^{(t+\frac{1}{2})} \gets \theta_k^{(t+1)} + s \lambda_k^{(t+1)}$
        \ENDFOR
        \STATE with probability $1 - p$ do global communication
        \STATE $\hspace{1.3em}$ client $k$ send $\theta_{k, 0}^{(t+\frac{1}{2})}$ to server $\forall k \in [K]$
        \STATE $\hspace{1.3em}$ {\bfseries Server Update:} $\theta^{(t+1)} \gets \frac{1}{K} \sum\limits_{k=1}^K \theta_{k, 0}^{(t+\frac{1}{2})}$ \COMMENT{compute global average}
        \STATE $\hspace{1.3em}$ Server broadcast $\theta^{(t+1)}$ to clients $k \in [K]$
        \STATE $\hspace{1.3em}$ On client $k:$ $\theta^{(t+1)}_{k,0} \gets \theta^{(t+1)}, ~ \forall k \in [K]$
        \STATE with probability $p$ skip global communication:
        \STATE $\hspace{1.3em}$ {\bfseries Client Update:} $\theta^{(t+1)}_{k,0} \gets \theta_{k, 0}^{(t+\frac{1}{2})}, ~ \forall k \in [K]$
        \STATE $\hspace{1.3em}$ \COMMENT{On server, $\theta^{(t+1)} \gets \theta^{(t)}$}
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
