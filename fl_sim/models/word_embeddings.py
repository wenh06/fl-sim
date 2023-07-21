"""
Adjusted from `TextAttack <https://github.com/QData/TextAttack>`_, and
`AI-Testing <https://git.openi.org.cn/Numbda/AI-Testing>`_.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Union, List, Optional, MutableMapping, Tuple, Dict, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm
from torch_ecg.utils import ReprMixin

from ..utils.const import CACHED_DATA_DIR
from ..utils._download_data import download_if_needed
from .tokenizers import WordLevelTokenizer, GloveTokenizer


__all__ = [
    "GloveEmbedding",
]


_CACHE_DIR = CACHED_DATA_DIR / "word_embeddings"


class AbstractWordEmbedding(ReprMixin, ABC):
    """Abstract class representing word embedding.

    This class specifies all the methods that is required to be defined
    so that it can be used for transformation and constraints.

    For custom word embedding, please create a
    class that inherits this class and implement the required methods.
    However, please first check if you can use `WordEmbedding` class,
    which has a lot of internal methods implemented.

    """

    __name__ = "AbstractWordEmbedding"

    @abstractmethod
    def __getitem__(self, index: Union[str, int]) -> np.ndarray:
        """Gets the embedding vector for word/ID.

        Parameters
        ----------
        index : str or int
            If `index` is a string,
            it is the word to be converted to embedding vector.
            If `index` is an integer,
            it is the ID of the word to be converted to embedding vector.

        Returns
        -------
        vector : numpy.ndarray
            1-D embedding vector.
            If corresponding vector cannot be found for `index`, returns ``None``.

        """
        raise NotImplementedError

    @abstractmethod
    def word2index(self, word: str) -> int:
        """Convert between word to id (i.e. index of word in embedding matrix).

        Parameters
        ----------
        word : str
            Word to be converted.

        Returns
        -------
        index : int
            Index of `word` in embedding matrix.

        """
        raise NotImplementedError

    @abstractmethod
    def index2word(self, index: int) -> str:
        """Convert index to corresponding word.

        Parameters
        ----------
        index : int
            Index of word in embedding matrix.

        Returns
        -------
        word : str
            Word corresponding to `index`.

        """
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        return []

    @property
    @abstractmethod
    def url(self) -> str:
        """URL of the embedding file."""
        raise NotImplementedError

    @property
    def cache_dir(self) -> Path:
        """Cache directory of the embedding file."""
        return _CACHE_DIR / Path(self.url).stem

    def download(self) -> None:
        """Download the embedding file if it is not already downloaded."""
        download_if_needed(url=self.url, dst_dir=self.cache_dir, extract=True)

    @abstractmethod
    def get_embedding_matrix(self) -> np.ndarray:
        """Get the embedding matrix."""
        raise NotImplementedError

    @property
    def dim(self) -> int:
        """Dimension of the embedding."""
        return self.get_embedding_matrix().shape[1]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the embedding."""
        return self.get_embedding_matrix().shape[0]

    def get_embedding_layer(
        self,
        oov: str = "<oov>",
        pad: str = "<pad>",
        freeze: bool = True,
    ) -> torch.nn.Embedding:
        """Returns a :class:`~torch.nn.Embedding` layer
        that is initialized with the embedding matrix.

        Parameters
        ----------
        oov : str
            The out-of-vocabulary token.
        pad : str
            The padding token.
        freeze : bool, default True
            Whether to freeze the embedding layer or not.

        Returns
        -------
        embedding_layer : torch.nn.Embedding
            Embedding layer initialized with the embedding matrix.

        """
        return EmbeddingLayer(
            embedding_matrix=self.get_embedding_matrix(),
            word2id=self._word2index,
            oov=oov,
            pad=pad,
        )

    def _get_tokenizer(self) -> WordLevelTokenizer:
        """Get the tokenizer."""
        embedding_layer = self.get_embedding_layer()
        tokenizer = WordLevelTokenizer(
            word_id_map=embedding_layer.word2id,
            oov_token=embedding_layer.oovid,
            pad_token=embedding_layer.padid,
        )
        del embedding_layer
        return tokenizer

    def get_input_tensor(
        self, input_sentences: Union[str, Sequence[str]]
    ) -> torch.Tensor:
        """Returns a :class:`~torch.Tensor`
        that can be used as input to the embedding layer.

        Parameters
        ----------
        input_sentences : str or Sequence[str]
            The input sentences.

        Returns
        -------
        input_tensor : torch.Tensor
            Input tensor that can be used as input to the embedding layer.

        """
        if isinstance(input_sentences, str):
            input_sentences = [input_sentences]
        if not hasattr(self, "_tokenizer") or self._tokenizer is None:
            self._tokenizer = self._get_tokenizer()
        return torch.tensor(
            [
                encoding.ids
                for encoding in self._tokenizer.encode_batch(
                    input_sentences, add_special_tokens=False
                )
            ]
        )

    @property
    def sizeof(self) -> str:
        """Size of the embedding matrix in human readable format."""
        nbytes = self._embedding_matrix.nbytes
        unit = {
            0: "B",
            1: "KB",
            2: "MB",
            3: "GB",
            4: "TB",
        }
        i = 0
        while nbytes >= 1024:
            nbytes /= 1024
            i += 1
        return f"{nbytes:.2f} {unit[i]}"


class WordEmbedding(AbstractWordEmbedding):
    """Object for loading word embeddings and related distances.

    Parameters
    ----------
    embedding_matrix : numpy.ndarray
        2-D array of shape ``N x D`` where
        ``N`` represents size of vocab and ``D`` is the dimension of embedding vectors.
    word2index : mapping
        Dictionary (or a similar object) that maps word to its index with
        in the embedding matrix.
    index2word : mapping
        Dictionary (or a similar object) that maps index to its word.
    device : torch.device, optional
        Device to use for nearest neighbour computation.
        Automatically set to ``cuda`` if available.

    """

    __name__ = "WordEmbedding"

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        word2index: MutableMapping,
        index2word: MutableMapping,
        device: Optional[torch.device] = None,
    ) -> None:
        self._embedding_matrix = embedding_matrix
        self._word2index = word2index
        self._index2word = index2word
        self.device = device

    def get_embedding_matrix(self) -> np.ndarray:
        return self._embedding_matrix

    def __getitem__(self, index: Union[str, int]) -> np.ndarray:
        if isinstance(index, str):
            try:
                index = self._word2index[index]
            except KeyError:
                return None
        try:
            return self._embedding_matrix[index]
        except IndexError:
            # word embedding ID out of bounds
            return None

    def word2index(self, word: str) -> int:
        return self._word2index[word]

    def index2word(self, index: int) -> str:
        return self._index2word[index]


class GloveEmbedding(WordEmbedding):
    """Class for loading GloVe word embeddings and related distances.

    Parameters
    ----------
    name : str, default "glove.6B.300d"
        Name of the GloVe embedding to load. Available options are:

            - glove.6B.50d
            - glove.6B.100d
            - glove.6B.200d
            - glove.6B.300d
            - glove.42B.300d
            - glove.840B.300d
            - glove.twitter.27B.25d
            - glove.twitter.27B.50d
            - glove.twitter.27B.100d
            - glove.twitter.27B.200d

    device : torch.device, optional
        Device to use for nearest neighbour computation.
        Automatically set to ``cuda`` if available.

    """

    __name__ = "GloveEmbedding"

    GLOVE_EMBEDDING_NAMES = {
        "glove.6B.50d": "glove.6B.50d.txt",
        "glove.6B.100d": "glove.6B.100d.txt",
        "glove.6B.200d": "glove.6B.200d.txt",
        "glove.6B.300d": "glove.6B.300d.txt",
        "glove.42B.300d": "glove.42B.300d.txt",
        "glove.840B.300d": "glove.840B.300d.txt",
        "glove.twitter.27B.25d": "glove.twitter.27B.25d.txt",
        "glove.twitter.27B.50d": "glove.twitter.27B.50d.txt",
        "glove.twitter.27B.100d": "glove.twitter.27B.100d.txt",
        "glove.twitter.27B.200d": "glove.twitter.27B.200d.txt",
    }

    def __init__(
        self,
        name: str = "glove.6B.300d",
        device: Optional[torch.device] = None,
    ) -> None:
        assert name in self.GLOVE_EMBEDDING_NAMES, (
            f"Invalid name: {name}. ",
            f"Available options are: {', '.join(self.GLOVE_EMBEDDING_NAMES)}",
        )
        self.name = name
        self.download()
        super().__init__(*self._load_embedding(), device=device)
        self._tokenizer = self._get_tokenizer()

    @staticmethod
    def get_url(name: str) -> str:
        """Get the URL for the GloVe embedding file."""
        # domain = "https://nlp.stanford.edu/data"
        domain = "https://huggingface.co/stanfordnlp/glove/resolve/main"
        return {
            "glove.6B.50d": f"{domain}/glove.6B.zip",
            "glove.6B.100d": f"{domain}/glove.6B.zip",
            "glove.6B.200d": f"{domain}/glove.6B.zip",
            "glove.6B.300d": f"{domain}/glove.6B.zip",
            "glove.42B.300d": f"{domain}/glove.42B.300d.zip",
            "glove.840B.300d": f"{domain}/glove.840B.300d.zip",
            "glove.twitter.27B.25d": f"{domain}/glove.twitter.27B.zip",
            "glove.twitter.27B.50d": f"{domain}/glove.twitter.27B.zip",
            "glove.twitter.27B.100d": f"{domain}/glove.twitter.27B.zip",
            "glove.twitter.27B.200d": f"{domain}/glove.twitter.27B.zip",
        }[name]

    @staticmethod
    def get_vocab_file(name: str = "glove.6B.300d") -> Path:
        """Get the path to the vocab file for the GloVe embedding file."""
        url = Path(GloveEmbedding.get_url(name))
        vocab_file = (
            _CACHE_DIR / url.stem / GloveEmbedding.GLOVE_EMBEDDING_NAMES[name]
        ).with_suffix(".vocab.json")
        return vocab_file

    @staticmethod
    def get_tokenizer(
        name: str = "glove.6B.300d", max_length: int = 256
    ) -> "GloveTokenizer":
        """Get the tokenizer for the GloVe embedding file.

        Parameters
        ----------
        name : str, default "glove.6B.300d"
            Name of the GloVe embedding to load. Available options are:

                - glove.6B.50d
                - glove.6B.100d
                - glove.6B.200d
                - glove.6B.300d
                - glove.42B.300d
                - glove.840B.300d
                - glove.twitter.27B.25d
                - glove.twitter.27B.50d
                - glove.twitter.27B.100d
                - glove.twitter.27B.200d

        max_length : int, default 256
            Maximum length of the input sequence.

        """
        vocab_file = GloveEmbedding.get_vocab_file(name)
        if not vocab_file.exists():
            raise FileNotFoundError(
                f"Vocab file not found at {vocab_file}. "
                "In this case, one should first instantiate a Glove Embedding object, "
                "which will download the embedding files to create the vocab file"
            )
        return GloveTokenizer(vocab_file=vocab_file, max_length=max_length)

    @property
    def url(self) -> str:
        return self.get_url(self.name)

    def _load_embedding(self) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
        """Load GloVe embeddings from file.

        Returns
        -------
        embedding_matrix : numpy.ndarray
            Embedding matrix of shape ``(num_embeddings, embedding_dim)``.
        word2index : Dict[str, int]
            Mapping from word to embedding ID.
        index2word : Dict[int, str]
            Mapping from embedding ID to word.

        """
        word2index = {}
        index2word = {}
        embedding_matrix = []
        total_lines = {
            "glove.6B.50d": 400001,  # update from 400000 to 400001, <unk> is added
            "glove.6B.100d": 400001,
            "glove.6B.200d": 400001,
            "glove.6B.300d": 400001,
            "glove.42B.300d": 1917495,  # by copilot, might be wrong
            "glove.840B.300d": 2196017,  # by copilot, might be wrong
            "glove.twitter.27B.25d": 1193514,  # by copilot, might be wrong
            "glove.twitter.27B.50d": 1193514,  # by copilot, might be wrong
            "glove.twitter.27B.100d": 1193514,  # by copilot, might be wrong
            "glove.twitter.27B.200d": 1193514,  # by copilot, might be wrong
        }[self.name]
        with open(self.file_path, "r") as f:
            with tqdm(
                f,
                desc=f"Loading {self.name} embeddings",
                total=total_lines,
                unit="lines",
                mininterval=1.0,
            ) as pbar:
                for i, line in enumerate(pbar):
                    line = line.strip().split()
                    word = line[0]
                    word2index[word] = i
                    index2word[i] = word
                    embedding_matrix.append([float(x) for x in line[1:]])
        embedding_matrix = np.array(embedding_matrix)
        return embedding_matrix, word2index, index2word

    def _get_tokenizer(self, max_length: int = 256) -> GloveTokenizer:
        try:
            tokenizer = GloveEmbedding.get_tokenizer(self.name)
        except FileNotFoundError:
            embedding_layer = self.get_embedding_layer()
            tokenizer = GloveTokenizer(
                vocab_file=GloveEmbedding.get_vocab_file(self.name),
                word_id_map=embedding_layer.word2id,
                pad_token_id=embedding_layer.padid,
                unk_token_id=embedding_layer.oovid,
                max_length=max_length,
            )
            del embedding_layer
        return tokenizer

    @property
    def file_path(self) -> Path:
        """File path of the GloVe embedding file."""
        return self.cache_dir / self.GLOVE_EMBEDDING_NAMES[self.name]

    def download(self) -> None:
        """Download the GloVe embedding file."""
        if not self.file_path.exists():
            super().download()

    def extra_repr_keys(self) -> List[str]:
        return ["name"]


class EmbeddingLayer(torch.nn.Module):
    """A layer of a model that replaces word IDs with their embeddings.

    This is a useful abstraction for any nn.module which wants to take word IDs
    (a sequence of text) as input layer but actually manipulate words' embeddings.

    Requires some pre-trained embedding with associated word IDs.

    Parameters
    ----------
    embedding_matrix : numpy.ndarray
        Embedding matrix of shape ``(num_embeddings, embedding_dim)``.
    word2id : Dict[str, int]
        Mapping from word to embedding ID.
    oov : str, default "<oov>"
        Out-of-vocabulary token.
    pad : str, default "<pad>"
        Padding token.
    freeze : bool, default True
        Whether to freeze the embedding layer or not.

    """

    __name__ = "EmbeddingLayer"

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        word2id: Dict[str, int],
        oov: str = "<oov>",
        pad: str = "<pad>",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.word2id = deepcopy(word2id)

        if oov not in word2id:
            self.word2id[oov] = len(self.word2id)

        if pad not in word2id:
            self.word2id[pad] = len(self.word2id)

        self.vocab_size, self.dim = len(self.word2id), embedding_dim
        self.oovid = self.word2id[oov]
        self.padid = self.word2id[pad]

        self.embedding = torch.nn.Embedding(self.vocab_size, self.dim)
        self.embedding.weight.data.uniform_(-0.25, 0.25)
        self.embedding.weight.data[:vocab_size].copy_(
            torch.from_numpy(embedding_matrix)
        )
        self.embedding.weight.requires_grad = not freeze

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embedding(input)
