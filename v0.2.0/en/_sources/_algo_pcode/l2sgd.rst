.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{L2SGD}}
    \begin{algorithmic}
    \REQUIRE penalty coefficient $\lambda,$ learning rate $\eta,$ probability $p \in (0, 1)$

    \STATE {Initiation:}
    \STATE local (clients) model parameters $\theta_k^{(0)} \in \R^d, ~ \forall k \in [K]$

    \FOR{each round $t = 0, 1, \cdots, T-1$}
    \STATE \COMMENT{with probability $1-p$: no global communication, only clients update}
    \STATE \FOR{each client $k \in [K]$ {in parallel}}
    \STATE sample $j \in [m]$ (uniformly at random)
    \STATE $g_k^{(t)} \gets \frac{1}{K(1-p)} \left( \nabla f_{k,j}(\theta_k^{(t)}) \right)$
    \STATE $\theta_k^{(t+1)} \gets \theta_k^{(t)} - \eta g_k^{(t)}$
    \STATE \ENDFOR

    \STATE \COMMENT{with probability $p$: global communication}
    \STATE client $k$ sends $\theta_{k}^{(t)}$ to the server, $\forall k \in [K]$
    \STATE {Server Update:}
    \STATE $\hspace{1.3em}$ $\theta^{(t)} \gets \frac{1}{K} \sum\limits_{k=1}^K \theta_{k}^{(t)}$ \COMMENT{compute the global average}
    \STATE server broadcasts $\theta^{(t)}$ to clients $k \in [K]$
    \STATE {Clients Update:}
    \STATE \FOR{each client $k \in [K]$ {in parallel}}
    \STATE $g_k^{(t)} \gets \frac{\lambda}{Kp} \left( \theta_k^{(t)} - \theta^{(t)} \right)$
    \STATE $\theta_k^{(t+1)} \gets \theta_k^{(t)} - \eta g_k^{(t)}$
    \STATE \ENDFOR
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
