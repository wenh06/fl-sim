.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{pFedMe}}
    \begin{algorithmic}
    \REQUIRE penalty coefficient $\lambda,$ learning rate $\eta,$ global update smoothing factor $\beta,$ {\bfseries proximal solvers} $\texttt{prox\_update}_k$

    \STATE {\bfseries Initiation:}
    \STATE $\hspace{1.3em}$ {\bfseries Init server:} global model parameters $\theta^{(0)} \in \mathbb{R}^d$
    \STATE $\hspace{1.3em}$ {\bfseries Init clients:} personalized model parameters $\theta_k^{(0)} \in \mathbb{R}^d, ~ \forall k \in [K]$

    \FOR{each round $t = 0, \cdots, T-1$}
        \STATE Server sends $\theta^{(t)}$ to all clients
        \FOR{each client $k = 1, \cdots, K$ in parallel}
            \STATE $\omega_{k}^{(t, 0)} = \theta^{(t)}$ \COMMENT{a copy of the global model $\theta^{(t)}$}
            \FOR{$r = 0,\cdots, R-1$}
                \COMMENT{solve inner problem $\min\limits_{\theta_k \in \mathbb{R}^d} \left\{ f_k(\theta_k) + \frac{\lambda}{2} \left\lVert \theta_k - \theta \right\rVert^2 \right\}$}
                \STATE sample a mini-batch $b_r$
                \COMMENT{use proximal solver $\texttt{prox\_update}_k$ to solve $\underset{\theta_k}{\text{argmin}} \left\{ \ell_k(\theta_k; b_r) + \frac{\lambda}{2} \left\lVert \theta_k - \omega_{k}^{(t, r)} \right\rVert^2 \right\}, \text{ where } f_k(\theta_k) = \mathbb{E}_{b}[\ell_k(\theta_k; b)]$}
                \STATE $\theta_k^{(t, r)} \gets \texttt{prox\_update}_k (\omega_{k}^{(t, r)}; b_r)$
                \COMMENT{update the local copy of the global model}
                \STATE $\omega_{k}^{(t, r+1)} \gets \omega_{k}^{(t, r)} - \eta \lambda \left( \omega_{k}^{(t, r)} - \theta_k^{(t, r)} \right)$
            \ENDFOR
            \STATE $\theta_k^{(t+1)} \gets \theta_k^{(t, R)}$ \COMMENT{update the personalized model}
        \ENDFOR
        \STATE Server uniformly samples a subset of clients $\mathcal{S}^{(t)}$
        \STATE each client in $\mathcal{S}^{(t)}$ sends the local model $\omega_{k}^{(t, R)}$ to the server
        \STATE Sever update $\theta^{(t+1)} \gets (1-\beta)\theta^{(t)} + \frac{\beta}{\# \mathcal{S}^{(t)}} \sum\limits_{k \in \mathcal{S}^{(t)}} \omega_{k}^{(t, R)}$
    \ENDFOR
    \STATE final global model: $\theta^* \gets \theta^{(T)}$
    \STATE final personalized models: $\theta_k^* \gets \theta_k^{(T)}$
    \end{algorithmic}
    \end{algorithm}
