.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{Pseudo-code for \texttt{FedAvg}}
    \begin{algorithmic}
    \REQUIRE learning rate $\eta$, batch size $B$

    \STATE {\bfseries Initiation:} global (server) model parameters $\theta^{(0)} \in \R^d$

    \FOR{each round $t = 0, 1, \cdots, T-1$}
        \STATE $\mathcal{S}^{(t)} \gets$ (random set of clients) $\subseteq [K]$
        \STATE broadcast $\theta^{(t)}$ to clients $k \in \mathcal{S}^{(t)}$
        \FOR{each client $k \in \mathcal{S}^{(t)}$ {\bfseries in parallel}}
            \STATE $\theta_k^{(t)} \gets$ {\bfseries ClientUpdate}$(k, \theta^{(t)})$
            \STATE send $\theta_k^{(t)}$ to server
        \ENDFOR
        \STATE {\bfseries Server Update:}
        \STATE $\hspace{1.3em}$ $\theta^{(t+1)} \gets \frac{1}{\lvert \mathcal{S}^{(t)} \rvert} \sum\limits_{k\in \mathcal{S}^{(t)}} \theta_k^{(t)}$
    \ENDFOR

    \PROCEDURE{ClientUpdate}{$(k, \theta)$} \COMMENT{on client $k$}
    \STATE $\mathcal{B} \gets$ (split $\mathcal{D}_k$ into batches of size $B$)
    \FOR{local step $r = 0, 1, \cdots, R-1$}
      \FOR{batch $b \in \mathcal{B}$}
        \STATE $\theta \gets \theta - \eta \nabla \ell_k(\theta; b)$ \COMMENT{SGD}
      \ENDFOR
    \ENDFOR
    \RETURN $\theta$
    \ENDPROCEDURE
    \end{algorithmic}
    \end{algorithm}
