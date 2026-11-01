.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{FedSplit}}
    \begin{algorithmic}
    \REQUIRE {\bfseries proximal solvers} $\texttt{prox\_update}_k: \mathbb{R}^d \to \mathbb{R}^d$

    \STATE {\bfseries Initiation:} parameters $\theta^{(0)} \in \mathbb{R}^d$

    \FOR{each round $t = 0, 1, \cdots$}
        \STATE broadcast $\theta^{(t)}$ to clients $k \in [K]$
        \FOR{each client $k = 1, \cdots, K$ {\bfseries in parallel}}
            \STATE $\theta_k^{(t)} \gets \theta^{(t)}$
            \STATE $\theta_k^{(t+1/2)} \gets$ $\texttt{prox\_update}_k(2\theta^{(t)} - \theta_k^{(t)})$ \COMMENT{local prox step}
            \STATE $\theta_k^{(t+1)} \gets$ $\theta_k^{(t)} + 2(\theta_k^{(t+1/2)} - \theta^{(t)})$ \COMMENT{local centering step}
            \STATE send $\theta_k^{(t+1)}$ to server
        \ENDFOR
        \STATE {\bfseries Server Update:}
        \STATE $\hspace{1.3em}$ $\theta^{(t+1)} \gets \frac{1}{K} \sum\limits_{k=1}^K \theta_k^{(t+1)}$ \COMMENT{global averaging}
        \IF{meet convergent criteria}
            \STATE $\theta^* \gets \theta^{(t+1)}$
            \STATE {\bfseries break}
        \ENDIF
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
