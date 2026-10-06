.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{Ditto}}
    \begin{algorithmic}
    \REQUIRE penalty coefficient $\mu,$ learning rate $\eta,$ methods $\textbf{UpdateGlobal}, \textbf{Aggregate}$

    \STATE {\bfseries Initiation:}
    \STATE $\hspace{1.3em}$ {\bfseries Init server:} global model parameters $\theta^{(0)} \in \mathbb{R}^d$
    \STATE $\hspace{1.3em}$ {\bfseries Init clients:} local model parameters $\omega_k^{(0)} \in \mathbb{R}^d, ~ \forall k \in [K]$

    \FOR{each round $t = 0, 1, \cdots, T-1$}
        \STATE $\mathcal{S}^{(t)} \gets$ (random set of clients) $\subseteq [K]$
        \STATE broadcast $\theta^{(t)}$ to clients $k \in \mathcal{S}^{(t)}$
        \FOR{each client $k \in \mathcal{S}^{(t)}$ {\bfseries in parallel}}
            \COMMENT{solve the local sub-problem of $G(\dot)$ inexactly starting from $\theta^{(t)}$ to obtain $\theta_k^{(t)}$}
            \STATE $\theta_k^{(t)} \gets \textbf{UpdateGlobal}(\theta^{(t)}, f_k)$
            \COMMENT{update the personalized model via solving the proximal problem}
            \STATE $\omega_k^{(t+1)} \gets \omega_k^{(t)} - \eta \left( \nabla f_k(\omega_k^{(t)}) + \mu (\omega_k^{(t)} - \theta^{(t)}) \right)$
            \STATE send $\Delta_k^{(t)} \gets \theta_k^{(t)} - \theta^{(t)}$ to server
        \ENDFOR
        \STATE {\bfseries Server Update:}
        \STATE $\hspace{1.3em}$ $\theta^{(t+1)} \gets \textbf{Aggregate} (\theta^{(t)}, \{ \Delta_k^{(t)} \}_{k \in \mathcal{S}^{(t)}})$
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
