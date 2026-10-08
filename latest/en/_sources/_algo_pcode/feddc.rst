.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{FedDC}}
    \begin{algorithmic}
    \REQUIRE number of clients $N,$ number of rounds $T,$ local iterations $K,$ learning rate $\eta,$ correction coefficient $\alpha$

    \STATE {Initiation:}
    \STATE $w^0$ on the server
    \STATE $\theta_i^0 = w^0$ and $h_i^0 = 0$ for all $i \in [N]$ on the clients

    \FOR{each round $t = 0, 1, \cdots, T-1$}
    \STATE the server downloads $w^t$ (and $g^t$, if any) to the clients
    \STATE \FOR{each client $i$ {in parallel}}
    \STATE $\theta_i^{t,0} = w^t$
    \FOR{$k = 0, 1, \cdots, K-1$}
    \STATE $\hspace{1.3em}$ $\theta_i^{t,k+1} = \theta_i^{t,k} - \eta \nabla L_i(\theta_i^{t,k}) - \eta \alpha \left( \theta_i^{t,k} + h_i^t - w^t \right) - \frac{1}{K} \left( g_i^t - g^t \right)$
    \STATE \ENDFOR
    \STATE $\theta_i^+ = \theta_i^{t,K}$
    \STATE $h_i^{t+1} = h_i^t + \left( \theta_i^+ - w^t \right)$
    \STATE $g_i^{t+1} = \theta_i^+ - w^t$
    \STATE upload $\theta_i^+ + h_i^{t+1}$ and $g_i^{t+1}$ to the server
    \STATE \ENDFOR
    \STATE {Server Update:}
    \STATE $w^{t+1} = \sum\limits_{i=1}^N \frac{\lvert D_i \rvert}{\lvert D \rvert} \left( \theta_i^+ + h_i^{t+1} \right)$
    \STATE $g^{t+1} = \sum\limits_{i=1}^N \frac{\lvert D_i \rvert}{\lvert D \rvert} g_i^{t+1}$
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
