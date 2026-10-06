.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{FedProx}}
    \begin{algorithmic}
    \REQUIRE penalty constant $\mu,$ constant $\gamma \in [0, 1]$

    \STATE {Initiation:}
    \STATE global model parameters $\theta^{(0)} \in \mathbb{R}^d$

    \FOR{each round $t = 0, 1, \cdots, T-1$}
    \STATE $\mathcal{S}^{(t)} \gets$ (random set of clients) $\subseteq [K]$
    \STATE broadcast $\theta^{(t)}$ to clients $k \in \mathcal{S}^{(t)}$
    \STATE \FOR{each client $k \in \mathcal{S}^{(t)}$ {in parallel}}
    \STATE find a $\gamma$-inexact solution $\theta_k^{(t)}$ to $\underset{\theta_k}{\text{argmin}} h_k(\theta_k, \theta^{(t)}) := f_k(\theta_k) + \frac{\mu}{2} \lVert \theta_k - \theta^{(t)} \rVert^2$
    \STATE \COMMENT{Definition of $\gamma$-inexactness: $\nabla h_k(\theta_k^*, \theta^{(t)}) \leqslant \gamma h_k(\theta_k, \theta^{(t)}),$ where $\theta_k^*$ is the exact solution to $h_k.$}
    \STATE send $\theta_k^{(t)}$ to server
    \STATE \ENDFOR

    \STATE {Server Update:}
    \STATE $\hspace{1.3em}$ $\theta^{(t+1)} \gets \frac{1}{\lvert \mathcal{S}^{(t)} \rvert} \sum\limits_{k\in \mathcal{S}^{(t)}} \theta_k^{(t)}$
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
