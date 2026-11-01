.. pcode::
    :linenos:
    :scopelines:

    \begin{algorithm}
    \caption{PseudoCode for \texttt{FedCR}}
    \begin{algorithmic}
    \REQUIRE number of clients $m,$ number of rounds $T,$ local steps $K,$ learning rate $\eta,$ CMI coefficient $\beta$; class-wise initialization $\mu_0^c = 0,$ $\Sigma_0^c = I$ for all classes $c$

    \FOR{each round $t = 1, 2, \cdots, T$}
    \STATE the server samples a subset of active clients $\mathcal{P}_t$ and sends $w^f$, $\mu_t^c$ and $\Sigma_t^c$ ($\forall c \in \mathcal{C}$) to each $i \in \mathcal{P}_t$
    \STATE \FOR{each client $i \in \mathcal{P}_t$ {in parallel}}
    \STATE initialize the local model as $w_{i,0} = [w^f, w_i^p]$
    \FOR{$k = 1, 2, \cdots, K$}
    \STATE $\hspace{1.3em}$ update the whole network by SGD on the local objective:
    \STATE $\hspace{1.3em}$ $w_{i,k} = w_{i,k-1} - \eta \nabla L_i$, where
    \STATE $\hspace{2.3em}$ $L_i \approx \frac{1}{N_i} \sum\limits_{n=1}^{N_i} \left[ -\log \hat{p}\left( y_i^{(n)} \,\middle|\, z^{(n)} \right) + \beta \, \mathrm{KL}\left[ p(z^{c = y_i^{(n)}} \mid x) \,\middle\|\, p(z \mid x_i^{(n)}) \right] \right]$
    \STATE \ENDFOR
    \STATE aggregate the class-wise local features via the local PoE: $\mu_i^c, \Sigma_i^c$ for $c \in \mathcal{C}_i$
    \STATE keep the personal head $w_i^p = w_{i,K}^p$
    \STATE send $w_{i,K}^f$ and $\mu_i^c, \Sigma_i^c$ ($\forall c \in \mathcal{C}_i$) to the server
    \STATE \ENDFOR
    \STATE {Server Update:}
    \STATE average the feature extractors: $w^f = \frac{1}{\lvert \mathcal{P}_t \rvert} \sum\limits_{i \in \mathcal{P}_t} w_{i,K}^f$
    \STATE for classes $c$ contained in no active client: keep $\mu_t^c = \mu_{t-1}^c$, $\Sigma_t^c = \Sigma_{t-1}^c$
    \STATE aggregate the global class-wise features via the PoE with prior $p(z) = \mathcal{N}(0, 1)$, i.e. for $c$ contained in some active client:
    \STATE $\hspace{1.3em}$ $\Sigma_t^c = \left( I + \sum\limits_{i \in \mathcal{P}_t} \left( \Sigma_i^c \right)^{-1} \right)^{-1}$, \quad $\mu_t^c = \Sigma_t^c \sum\limits_{i \in \mathcal{P}_t} \mu_i^c \left( \Sigma_i^c \right)^{-1}$
    \STATE send $w^f$, $\mu_t^c$ and $\Sigma_t^c$ ($\forall c \in \mathcal{C}$) to the clients
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
