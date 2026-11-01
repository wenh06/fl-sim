.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{APFL}}
    \begin{algorithmic}
    \REQUIRE mixture weights $\alpha_1, \ldots, \alpha_K,$ synchronization gap $\tau$

    \STATE {Initiation:}
        \STATE $\hspace{1.3em}$ {Init server:} global model parameters $\theta^{(0)} \in \mathbb{R}^d,$ random set of clients $S^{(t)} \subseteq [K]$
        \STATE $\hspace{1.3em}$ {Init clients:} local model parameters $\omega_k^{(0)} \in \mathbb{R}^d, ~ \theta_k^{(0)} \gets \theta^{(0)}, ~ \forall k \in [K]$
        \STATE $\hspace{1.3em}$ {Constants:} condition number $\kappa \gets \frac{L}{\mu},$ $a \gets \max\{128\kappa, \tau\}$

    \FOR{each round $t = 0, 1, \cdots, T-1$}
        \FOR{each client $k \in \mathcal{S}^{(t)}$ {in parallel}}
            \STATE $\bar{\omega}_k^{(t)} \gets \alpha_k \omega_k^{(t)} + (1-\alpha_k) \theta_k^{(t)}$ \COMMENT{mixture model}
            \STATE $\eta^{(t)} \gets \frac{16}{\mu(t+a)}$ \COMMENT{decay learning rate}
            \STATE $\theta_k^{(t+1)} \gets \theta_k^{(t)} - \eta^{(t)} \nabla f_k(\theta_k^{(t)})$ \COMMENT{inner problem (global model) update}
            \STATE $\omega_k^{(t+1)} \gets \omega_k^{(t)} - \eta^{(t)} \nabla f_k(\bar{\omega}_k^{(t)})$ \COMMENT{outer problem (personalized model) update}
            \COMMENT{Optional: adaptive mixture weights update}
            \COMMENT{$\alpha_k \gets \alpha_k - \eta^{(t)} \nabla_{\alpha_k} f_k(\bar{\omega}_k^{(t)}) = \alpha_k - \eta^{(t)} \left\langle \omega_k^{(t)} - \theta_k^{(t)}, \nabla f_k(\bar{\omega}_k^{(t)}) \right\rangle$}
        \ENDFOR

        \IF{$t$ not divides synchronization gap $\tau$}
            \STATE {Server updates:} $\mathcal{S}^{(t+1)} \gets \mathcal{S}^{(t)}$
        \ELSE
            \STATE each client $k \in \mathcal{S}^{(t)}$ sends $\theta_k^{(t+1)}$ to server
            \STATE {Server updates:}
                \STATE $\hspace{1.3em}$ $\theta^{(t+1)} \gets \frac{1}{\# \mathcal{S}^{(t)}} \sum_{k \in \mathcal{S}^{(t)}} \theta_k^{(t+1)}$
                \STATE $\hspace{1.3em}$ $\mathcal{S}^{(t+1)} \gets$ (random set of clients) $\subseteq [K]$
                \STATE $\hspace{1.3em}$ broadcast $\theta^{(t+1)}$ to clients $k \in S^{(t+1)}:$ $\theta_k^{(t+1)} \gets \theta^{(t+1)}$
        \ENDIF
    \ENDFOR

    \STATE final personalized model: $\hat{\omega}_k \gets \frac{1}{S_T}\sum\limits_{t=1}^{T} p_t \left( \alpha_k \omega_k^{(t)} + (1-\alpha_k)\frac{1}{\#\mathcal{S}^{(t-1)}}\sum\limits_{k\in\mathcal{S}^{(t-1)}}\theta_k^{(t)} \right)$
    \STATE final global model: $\hat{\theta} \gets \frac{1}{S_T}\sum\limits_{t=1}^{T} \frac{p_t}{\#\mathcal{S}^{(t-1)}} \sum\limits_{k\in\mathcal{S}^{(t-1)}}\theta_k^{(t)}$
    \STATE where $p_t = (t+a)^2, S_T = \sum\limits_{t=1}^{T}p_t$
    \end{algorithmic}
    \end{algorithm}
