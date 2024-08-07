"""
simple neural network models
"""

import re
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F  # noqa: F401
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch_ecg.models._nets import get_activation
from torch_ecg.utils import SizeMixin, get_kwargs
from torchvision.models.resnet import BasicBlock, ResNet, conv1x1, resnet18

from .utils import CLFMixin, DiffMixin, REGMixin

__all__ = [
    "MLP",
    "FedPDMLP",
    "CNNMnist",
    "CNNFEMnist",
    "CNNFEMnist_Tiny",
    "CNNCifar",
    "CNNCifar_Small",
    "CNNCifar_Tiny",
    "RNN_OriginalFedAvg",
    "RNN_StackOverFlow",
    "RNN_Sent140",
    "RNN_Sent140_LITE",
    "ResNet18",
    "ResNet10",
    "LogisticRegression",
    "SVC",
    "SVR",
    "ShrinkedResNet",
]


class MLP(nn.Sequential, CLFMixin, SizeMixin, DiffMixin):
    """Multi-layer perceptron.

    can be used for

    1. logistic regression (for classification) using cross entropy loss (CrossEntropyLoss, BCEWithLogitsLoss, etc)
    2. regression (for regression) using MSE loss
    3. SVM (for classification) using hinge loss (MultiMarginLoss, MultiLabelMarginLoss, etc)
    4. etc.

    Parameters
    ----------
    dim_in : int
        Number of input features.
    dim_out : int
        Number of output features.
    dim_hidden : int or List[int], optional
        Number of hidden features.
        If is None, then no hidden layer.
    activation : str or List[str], default "relu"
        Activation function(s) for hidden layers.
    dropout : float or List[float], default 0.2
        Dropout rate(s) for hidden layers.
    ndim : int, default 2
        Number of dimensions of input data.
        2 for image data, 1 for sequence data,
        0 for vectorized data.

    """

    __name__ = "MLP"

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: Optional[Union[int, Sequence[int]]] = None,
        activation: Union[str, Sequence[str]] = "relu",
        dropout: Union[float, Sequence[float]] = 0.2,
        ndim: int = 2,
    ) -> None:
        super().__init__()
        self.ndim = ndim
        if ndim == 2:
            self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        elif ndim == 1:
            self.add_module("flatten", Rearrange("b c l -> b (c l)"))
        elif ndim == 0:
            pass
        else:
            raise ValueError(f"ndim must be 0, 1 or 2, got {ndim}")
        dims = []
        if dim_hidden is not None:
            if isinstance(dim_hidden, int):
                dims = [
                    dim_hidden,
                ]
            else:
                dims = list(dim_hidden)
        if isinstance(activation, (str, type(None))):
            activation = [activation for _ in range(len(dims))]
        if isinstance(dropout, (float, int)):
            dropout = [dropout for _ in range(len(dims))]
        for i, dim in enumerate(dims):
            self.add_module(f"linear_{i+1}", nn.Linear(dim_in, dim))
            if activation[i] is not None:
                self.add_module(f"activation_{i+1}", get_activation(activation[i], kw_act={}))
            if dropout[i] > 0:
                self.add_module(f"dropout_{i+1}", nn.Dropout(dropout[i]))
            dim_in = dim
        self.add_module(f"linear_{len(dims)+1}", nn.Linear(dim_in, dim_out))


class FedPDMLP(nn.Sequential, CLFMixin, SizeMixin, DiffMixin):
    """Multi-layer perceptron modified from
    `FedPD/models.py <https://github.com/564612540/FedPD>`_.

    Parameters
    ----------
    dim_in : int
        Number of input features.
    dim_hidden : int
        Number of hidden features.
    dim_out : int
        Number of output features.
    ndim : int, default 2
        Number of dimensions of input data.
        2 for image data, 1 for sequence data,
        0 for vectorized data.

    """

    __name__ = "FedPDMLP"

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, ndim: int = 2) -> None:
        """ """
        super().__init__()
        if ndim == 2:
            self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        elif ndim == 1:
            self.add_module("flatten", Rearrange("b c l -> b (c l)"))
        elif ndim == 0:
            pass
        else:
            raise ValueError(f"ndim must be 0, 1 or 2, got {ndim}")
        self.add_module("layer_input", nn.Linear(dim_in, dim_hidden))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("dropout", nn.Dropout(p=0.2, inplace=True))
        self.add_module("layer_hidden", nn.Linear(dim_hidden, dim_out))


class CNNMnist(nn.Sequential, CLFMixin, SizeMixin, DiffMixin):
    """Convolutional neural network using MNIST type input.

    Modified from `FedPD/models.py <https://github.com/564612540/FedPD>`_.

    Input size: ``(batch_size, 1, 28, 28)``.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    """

    __name__ = "CNNMnist"

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.add_module("conv1", nn.Conv2d(1, 10, kernel_size=5))
        self.add_module("mp1", nn.MaxPool2d(2))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(10, 20, kernel_size=5))
        self.add_module("drop1", nn.Dropout2d(p=0.2, inplace=True))
        self.add_module("mp2", nn.MaxPool2d(2))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        self.add_module("fc1", nn.Linear(320, 50))
        self.add_module("relu3", nn.ReLU(inplace=True))
        self.add_module("drop2", nn.Dropout(p=0.2, inplace=True))
        self.add_module("fc2", nn.Linear(50, num_classes))


class CNNFEMnist(nn.Sequential, CLFMixin, SizeMixin, DiffMixin):
    """Convolutional neural network using FEMnist type input.

    Modified from `FedPD/models.py <https://github.com/564612540/FedPD>`_.

    Input shape: ``(batch_size, 1, 28, 28)``.

    Parameters
    ----------
    num_classes : int, default 62
        Number of output classes.

    """

    __name__ = "CNNFEMnist"

    def __init__(self, num_classes: int = 62) -> None:
        super().__init__()
        in_channels = 1
        for i, out_channels in enumerate([32, 64]):
            self.add_module(
                f"conv_block{i+1}",
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ),
            )
            in_channels = out_channels
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        mlp_mid_channels = 2048
        self.add_module(
            "mlp",
            nn.Sequential(
                nn.Linear(7 * 7 * in_channels, mlp_mid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_mid_channels, num_classes),
            ),
        )


class CNNFEMnist_Tiny(nn.Sequential, CLFMixin, SizeMixin, DiffMixin):
    """Tiny version of :class:`CNNFEMnist`.

    Modified from `FedPD/models.py <https://github.com/564612540/FedPD>`_.

    Input shape: ``(batch_size, 1, 28, 28)``.

    Parameters
    ----------
    num_classes : int, default 62
        Number of output classes.

    """

    __name__ = "CNNFEMnist_Tiny"

    def __init__(self, num_classes: int = 62) -> None:
        super().__init__()
        in_channels = 1
        for i, out_channels in enumerate([16, 32]):
            self.add_module(
                f"conv_block{i+1}",
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ),
            )
            in_channels = out_channels
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        mlp_mid_channels = 256
        self.add_module(
            "mlp",
            nn.Sequential(
                nn.Linear(7 * 7 * in_channels, mlp_mid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_mid_channels, num_classes),
            ),
        )


class CNNCifar(nn.Sequential, CLFMixin, SizeMixin, DiffMixin):
    """Convolutional neural network using CIFAR type input.

    Input shape: ``(batch_size, 3, 32, 32)``.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    """

    __name__ = "CNNCifar"

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        in_channels = 3
        for idx, out_channels in enumerate([64, 128]):
            self.add_module(
                f"conv_block{idx+1}",
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=5),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ),
            )
            in_channels = out_channels
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        mlp_mid_channels = 384
        self.add_module(
            "mlp",
            nn.Sequential(
                nn.Linear(out_channels * 5 * 5, mlp_mid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_mid_channels, num_classes),
            ),
        )


class CNNCifar_Small(nn.Sequential, CLFMixin, SizeMixin, DiffMixin):
    """Convolutional neural network using CIFAR type input.

    Input shape: ``(batch_size, 3, 32, 32)``, smaller
    in size than `CNNCifar`

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    """

    __name__ = "CNNCifar_Small"

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        in_channels = 3
        for idx, out_channels in enumerate([32, 84]):
            self.add_module(
                f"conv_block{idx+1}",
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=5),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ),
            )
            in_channels = out_channels
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        mlp_mid_channels = 256
        self.add_module(
            "mlp",
            nn.Sequential(
                nn.Linear(out_channels * 5 * 5, mlp_mid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_mid_channels, num_classes),
            ),
        )


class CNNCifar_Tiny(nn.Sequential, CLFMixin, SizeMixin, DiffMixin):
    """Convolutional neural network using CIFAR type input.

    Modified from `FedPD/models.py <https://github.com/564612540/FedPD>`_.

    Input shape: ``(batch_size, 3, 32, 32)``.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    NOTE
    ----
    This model is too tiny.

    """

    __name__ = "CNNCifar_Tiny"

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.add_module(
            "conv_block1",
            nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ),
        )
        self.add_module(
            "conv_block2",
            nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ),
        )
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        self.add_module(
            "mlp",
            nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(inplace=True),
                nn.Linear(120, 84),
                nn.ReLU(inplace=True),
                nn.Linear(84, num_classes),
            ),
        )


class RNN_OriginalFedAvg(nn.Module, CLFMixin, SizeMixin, DiffMixin):
    """Creates a RNN model using LSTM layers for
    Shakespeare language models (next character prediction task).

    This replicates the model structure in the paper [:footcite:ct:`mcmahan2017fed_avg`].
    This is also recommended model by [:footcite:ct:`reddi2020fed_opt`].

    Modified from `FedML <https://github.com/FedML-AI/FedML>`_.

    Parameters
    ----------
    embedding_dim : int
        The size of each embedding vector.
    vocab_size : int
        The number of different characters that can appear in the input.
    hidden_size : int
        The number of features in the hidden state h.


    .. footbibliography::

    """

    __name__ = "RNN_OriginalFedAvg"

    def __init__(self, embedding_dim: int = 8, vocab_size: int = 90, hidden_size: int = 256) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.rearrange = Rearrange("batch length vocab -> batch vocab length")

    def forward(self, input_seq: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input_seq : torch.Tensor
            Shape ``(batch_size, seq_len)``, dtype :class:`torch.long`.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, vocab_size, seq_len)``, dtype :class:`torch.float32`.

        """
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        # final_hidden_state = lstm_out[:, -1]
        # output = self.fc(final_hidden_state)
        # For fed_shakespeare
        output = self.rearrange(self.fc(lstm_out))
        return output

    def pipeline(
        self,
        truncated_sentence: str,
        char_to_id: Optional[Union[Callable[[str], int], Dict[str, int]]] = None,
        id_to_char: Optional[Union[Callable[[int], str], Dict[int, str]]] = None,
    ) -> str:
        """Predict the next character given a truncated sentence.

        Parameters
        ----------
        truncated_sentence : str
            The truncated sentence.
        char_to_id : Callable[[str], int] or Dict[str, int], optional
            A function that maps a character to its id.
        id_to_char : Callable[[int], str] or Dict[int, str], optional
            A function that maps an id to its character.

        Returns
        -------
        str
            The predicted next characters.

        """
        prev = self.training
        self.eval()
        assert char_to_id is not None or id_to_char is not None, "Either `char_to_id` or `id_to_char` must be provided."
        if char_to_id is None:
            assert isinstance(id_to_char, dict), "`id_to_char` must be a dict if `char_to_id` is not provided."
            char_to_id = {v: k for k, v in id_to_char.items()}
            id_to_char = id_to_char.get
            char_to_id = char_to_id.get
        elif id_to_char is None:
            assert isinstance(char_to_id, dict), "`char_to_id` must be a dict if `id_to_char` is not provided."
            id_to_char = {v: k for k, v in char_to_id.items()}
            char_to_id = char_to_id.get
            id_to_char = id_to_char.get
        else:  # both are provided
            if isinstance(char_to_id, dict):
                char_to_id = char_to_id.get
            if isinstance(id_to_char, dict):
                id_to_char = id_to_char.get
        with torch.no_grad():
            input_seq = torch.tensor([[char_to_id(c) for c in truncated_sentence]], dtype=torch.long)
            output = self(input_seq)
            predicted_ids = torch.argmax(output, dim=-1)
        self.train(prev)
        # return id_to_char(predicted_id.item())
        return "".join([id_to_char(idx.item()) for idx in predicted_ids[0]])


class RNN_StackOverFlow(nn.Module, CLFMixin, SizeMixin, DiffMixin):
    """Creates a RNN model using LSTM layers for StackOverFlow (next word prediction task).

    This replicates the model structure in the paper [:footcite:ct:`reddi2020fed_opt`].

    Modified from `FedML <https://github.com/FedML-AI/FedML>`_.

    Parameters
    ----------
    vocab_size : int, default 10000
        The number of different words that can appear in the input.
    num_oov_buckets : int, default 1
        The number of out-of-vocabulary buckets.
    embedding_size : int, default 96
        The size of each embedding vector.
    latent_size : int, default 670
        The number of features in the hidden state h.
    num_layers : int, default 1
        The number of recurrent layers (:class:`torch.nn.LSTM`).


    .. footbibliography::

    """

    __name__ = "RNN_StackOverFlow"

    def __init__(
        self,
        vocab_size: int = 10000,
        num_oov_buckets: int = 1,
        embedding_size: int = 96,
        latent_size: int = 670,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
        self.word_embeddings = nn.Embedding(
            num_embeddings=extended_vocab_size,
            embedding_dim=embedding_size,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=latent_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(latent_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, extended_vocab_size)
        self.rearrange = Rearrange("batch length vocab -> batch vocab length")

    def forward(self, input_seq: Tensor, hidden_state: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input_seq : torch.Tensor
            Shape ``(batch_size, seq_len)``, dtype :class:`torch.long`.
        hidden_state : torch.Tensor, optional
            Shape ``(num_layers, batch_size, latent_size)``, dtype :class:`torch.float32`.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, extended_vocab_size, seq_len)``, dtype :class:`torch.float32`.

        """
        embeds = self.word_embeddings(input_seq)
        self.lstm.flatten_parameters()
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        fc1_output = self.fc1(lstm_out)
        output = self.rearrange(self.fc2(fc1_output))
        return output

    def pipeline(
        self,
        truncated_sentence: str,
        word_to_id: Optional[Union[Callable[[str], int], Dict[str, int]]] = None,
        id_to_word: Optional[Union[Callable[[int], str], Dict[int, str]]] = None,
    ) -> str:
        """Predict the next word given a truncated sentence.

        Parameters
        ----------
        truncated_sentence : str
            The truncated sentence.
        word_to_id : Callable[[str], int] or Dict[str, int], optional
            A function that maps a word to its id.
        id_to_word : Callable[[int], str] or Dict[int, str], optional
            A function that maps an id to its word.

        Returns
        -------
        str
            The predicted next word.

        """
        prev = self.training
        self.eval()
        assert word_to_id is not None or id_to_word is not None, "Either `word_to_id` or `id_to_word` must be provided."
        if word_to_id is None:
            assert isinstance(id_to_word, dict), "`id_to_word` must be a dict if `word_to_id` is not provided."
            word_to_id = {v: k for k, v in id_to_word.items()}
            id_to_word = id_to_word.get
            word_to_id = word_to_id.get
        elif id_to_word is None:
            assert isinstance(word_to_id, dict), "`word_to_id` must be a dict if `id_to_word` is not provided."
            id_to_word = {v: k for k, v in word_to_id.items()}
            word_to_id = word_to_id.get
            id_to_word = id_to_word.get
        else:  # both are provided
            if isinstance(word_to_id, dict):
                word_to_id = word_to_id.get
            if isinstance(id_to_word, dict):
                id_to_word = id_to_word.get
        with torch.no_grad():
            input_seq = torch.tensor([[word_to_id(w) for w in truncated_sentence.split()]], dtype=torch.long)
            output = self(input_seq)
            predicted_ids = torch.argmax(output, dim=-1)
        self.train(prev)
        # return id_to_word(predicted_id.item())
        return " ".join([id_to_word(idx.item()) for idx in predicted_ids[0]])


class RNN_Sent140(nn.Module, CLFMixin, SizeMixin, DiffMixin):
    """Stacked :class:`~torch.nn.LSTM` model for sentiment analysis
    on the ``Sent140`` dataset.

    Adapted from FedProx/flearn/models/sent140/stacked_lstm.py [#fedprox]_.

    Parameters
    ----------
    latent_size : int, default 100
        The number of features in the hidden state h.
    num_classes : int, default 2
        The number of output classes.
    num_layers : int, default 2
        The number of recurrent layers (:class:`~torch.nn.LSTM`).
    embedding : str or :class:`GloveEmbedding`, default "glove.6B.50d"
        The name of the pre-trained GloVe embedding to use or a
        :class:`GloveEmbedding` object.

    References
    ----------
    .. [#fedprox] https://github.com/litian96/FedProx

    """

    __name__ = "RNN_Sent140"

    def __init__(
        self,
        latent_size: int = 100,
        num_classes: int = 2,
        num_layers: int = 2,
        embedding: Union[str, object] = "glove.6B.50d",
    ) -> None:
        from .word_embeddings import GloveEmbedding

        super().__init__()
        if isinstance(embedding, str):
            self.embedding_name = embedding
            glove_embedding = GloveEmbedding(self.embedding_name)
            self.word_embeddings = glove_embedding.get_embedding_layer(freeze=True)
            self.tokenizer = glove_embedding.get_tokenizer()
            del glove_embedding
        elif isinstance(embedding, GloveEmbedding):
            self.embedding_name = embedding.name
            self.word_embeddings = embedding.get_embedding_layer(freeze=True)
            self.tokenizer = embedding.get_tokenizer()
        else:
            raise TypeError(f"embedding must be a `str` or `GloveEmbedding`, got {type(embedding)}")
        self.lite_model = RNN_Sent140_LITE(latent_size, num_classes, num_layers, self.word_embeddings.dim)

    def forward(self, input_seq: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input_seq : torch.Tensor
            Shape ``(batch_size, seq_len)``, dtype :class:`torch.long`.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, num_classes)``, dtype :class:`torch.float32`.

        """
        embeds = self.word_embeddings(input_seq)  # shape: (batch_size, seq_len, embedding_dim)
        return self.lite_model(embeds)

    def pipeline(self, sentence: str) -> int:
        """Predict the sentiment of a sentence.

        Parameters
        ----------
        sentence : str
            The sentence to predict the sentiment of.

        Returns
        -------
        int
            The class index of the predicted sentiment.

        """
        prev = self.training
        self.eval()
        with torch.no_grad():
            input_seq = torch.tensor([self.tokenizer.encode(sentence)], dtype=torch.long)
            output = self(input_seq)
            predicted_class = torch.argmax(output, dim=-1)
        self.train(prev)
        return predicted_class.item()


class RNN_Sent140_LITE(nn.Module, CLFMixin, SizeMixin, DiffMixin):
    """Stacked :class:`~torch.nn.LSTM` model for sentiment analysis
    on the ``Sent140`` dataset.

    The same as :class:`RNN_Sent140` but without the GloVe embedding layer.
    Hence the inputs of this model are embedd vectors.

    Parameters
    ----------
    latent_size : int, default 100
        The number of features in the hidden state h.
    num_classes : int, default 2
        The number of output classes.
    num_layers : int, default 2
        The number of recurrent layers (:class:`torch.nn.LSTM`).
    embed_dim : int, default 50
        The dimension of the input embeddings.

    """

    __name__ = "RNN_Sent140_LITE"

    def __init__(
        self,
        latent_size: int = 100,
        num_classes: int = 2,
        num_layers: int = 2,
        embed_dim: int = 50,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=latent_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(latent_size, 300)
        # final binary classification layer
        self.fc2 = nn.Linear(300, num_classes)

    def forward(self, embeds: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        embeds : torch.Tensor
            Shape ``(batch_size, seq_len, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, num_classes)``.

        """
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embeds)
        final_hidden_state = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)
        fc1_output = self.fc1(final_hidden_state)  # shape: (batch_size, 30)
        output = self.fc2(fc1_output)  # shape: (batch_size, 1)
        return output


class ResNet18(ResNet, CLFMixin, SizeMixin, DiffMixin):
    """ResNet18 model for image classification.

    Parameters
    ----------
    num_classes : int
        The number of output classes.
    pretrained : bool, default False
        If ``True``, use a pretrained model from ``torchvision.models``.

    """

    __name__ = "ResNet18"

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        """ """
        super().__init__(
            BasicBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
        )
        if pretrained:
            if "weights" in get_kwargs(resnet18):
                from torchvision.models.resnet import ResNet18_Weights

                _model = resnet18(weights=ResNet18_Weights.DEFAULT)
            else:  # older versions of torchvision
                _model = resnet18(pretrained=True)
            self.load_state_dict(
                {k: v for k, v in _model.state_dict().items() if not re.findall("^fc\\.", k)},
                strict=False,
            )
            del _model


class ResNet10(ResNet, CLFMixin, SizeMixin, DiffMixin):
    """ResNet10 model for image classification.

    Parameters
    ----------
    num_classes : int
        The number of output classes.
    pretrained : bool, default False
        Not used. ResNet10 has no pretrained model.
        This parameter is only for compatibility with other models.

    """

    __name__ = "ResNet10"

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        """ """
        super().__init__(
            BasicBlock,
            [1, 1, 1, 1],
            num_classes=num_classes,
        )
        if pretrained:
            raise NotImplementedError("ResNet10 has no pretrained model.")


class LogisticRegression(nn.Module, CLFMixin, SizeMixin, DiffMixin):
    """Logistic regression model for classification task.

    Parameters
    ----------
    num_features : int
        The number of input features.
    num_classes : int
        The number of output classes.

    """

    __name__ = "LogisticRegression"

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.mlp = MLP(dim_in=num_features, dim_out=num_classes, ndim=0)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class SVC(nn.Module, CLFMixin, SizeMixin, DiffMixin):
    """Support vector machine classifier.

    Parameters
    ----------
    num_features : int
        The number of input features.
    num_classes : int
        The number of output classes.

    """

    __name__ = "SVC"

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.mlp = MLP(dim_in=num_features, dim_out=num_classes, ndim=0)
        self.criterion = nn.MultiMarginLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class SVR(nn.Module, REGMixin, SizeMixin, DiffMixin):
    """Support vector machine regressor.

    Parameters
    ----------
    num_features : int
        The number of input features.

    """

    __name__ = "SVR"

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.mlp = MLP(dim_in=num_features, dim_out=1, ndim=0)
        self.criterion = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class ShrinkedResNet(nn.Sequential, CLFMixin, SizeMixin, DiffMixin):
    """Shrinked version of ResNet so that the number of macro blocks can be customized (shrinked).

    Parameters
    ----------
    layers : List[int]
        Number of blocks in each macro block.
    num_classes : int, default 1000
        The number of output classes.

    """

    def __init__(
        self,
        layers: List[int],
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.add_module("conv1", nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False))
        self.add_module("bn1", self._norm_layer(self.inplanes))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        inplanes = self.inplanes
        for idx, num_blocks in enumerate(layers):
            stride = 1 if idx == 0 else 2
            self.add_module(
                f"layer{idx + 1}",
                self._make_layer(
                    inplanes,
                    num_blocks,
                    stride=stride,
                ),
            )
            inplanes *= 2
        self.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        self.add_module("fc", nn.Linear(self.inplanes, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        block = BasicBlock
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
