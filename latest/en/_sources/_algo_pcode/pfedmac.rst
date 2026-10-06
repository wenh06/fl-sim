.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{pFedMac}}
    \begin{algorithmic}
    \REQUIRE learning rate $\eta$, penalty coefficient $\lambda$, $\beta$

    \STATE {\bfseries Initiation:} global (server) model parameters $\theta^{(0)} \in \mathbb{R}^d$

    \FOR{each round $t = 0, 1, \cdots, T-1$}
        \STATE $\mathcal{S}^{(t)} \gets$ (random set of clients) $\subseteq [K]$
        \STATE broadcast $\theta^{(t)}$ to clients $k \in \mathcal{S}^{(t)}$
        \FOR{each client $k \in \mathcal{S}^{(t)}$ {\bfseries in parallel}}
            \STATE $\theta_k^{(t)} \gets$ {\bfseries ClientUpdate}$(k, \theta^{(t)})$
            \STATE send $\theta_k^{(t)}$ to server
        \ENDFOR
        \STATE {\bfseries Server Update:}
        \STATE $\hspace{1.3em}$ $\theta^{(t+1)} \gets (1 - \beta) \theta^{(t)} + \frac{\beta}{\lvert \mathcal{S}^{(t)} \rvert} \sum\limits_{k\in \mathcal{S}^{(t)}} \theta_k^{(t)}$
    \ENDFOR

    \PROCEDURE{ClientUpdate}{$(k, \theta)$} \COMMENT{on client $k$}
    \STATE $\omega_k^{(t,0)} = \theta_k^{(t,0)} = \theta^{(t)}$
    \FOR{local step $r = 0, 1, \cdots, R-1$}
      \STATE $\mathcal{D}_{k, r} \gets$ (sample a mini-batch data)
      \STATE $\omega_k^{(t,r)} \gets \argmin_{\omega_k} \left\{ \ell_k(\omega_k; \mathcal{D}_{k, r}) - \lambda \langle \omega_k, \theta_k^{(t,r)} \rangle \right\}$
      \STATE $\theta_k^{(t,r+1)} \gets \theta_k^{(t,r)} - \eta\lambda \left( \theta_k^{(t,r)} - \omega_k^{(t,r)} \right)$
    \ENDFOR
    \RETURN{$\theta_k^{(t,R)}$}
    \ENDPROCEDURE
    \end{algorithmic}
    \end{algorithm}
