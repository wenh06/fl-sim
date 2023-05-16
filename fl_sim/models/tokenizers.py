"""
Adjusted from
1. [TextAttack](https://github.com/QData/TextAttack)
2. [AI-Testing](https://git.openi.org.cn/Numbda/AI-Testing)
"""

import json
import re
import tempfile
import warnings
from pathlib import Path
from string import punctuation
from typing import Optional, Sequence, List, Union, Dict

import torch

try:
    from nltk.tokenize import word_tokenize as nltk_word_tokenize
    from nltk import download as nltk_download
    from nltk.data import find as nltk_find
except ModuleNotFoundError:
    nltk_word_tokenize = None
    nltk_download = None
    nltk_find = None
    warnings.warn("Package `nltk` is not installed, using naive tokenizer instead.")

try:
    import tokenizers as hf_tokenizers
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install the `tokenizers` package first. "
        "You can install it by running `pip install tokenizers`."
    )


__all__ = [
    "WordLevelTokenizer",
    "GloveTokenizer",
]


if nltk_word_tokenize is not None:
    # download necessary nltk data if not downloaded
    try:
        nltk_find("tokenizers/punkt")
    except LookupError:
        nltk_download("punkt", quiet=True)
    try:
        nltk_find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk_download("averaged_perceptron_tagger", quiet=True)
    try:
        nltk_find("taggers/universal_tagset")
    except LookupError:
        nltk_download("universal_tagset", quiet=True)
    try:
        nltk_find("corpora/wordnet")
    except LookupError:
        nltk_download("wordnet", quiet=True)
    try:
        nltk_find("corpora/stopwords")
    except LookupError:
        nltk_download("stopwords", quiet=True)
    try:
        nltk_find("corpora/omw")
    except LookupError:
        nltk_download("omw", quiet=True)


class WordLevelTokenizer(hf_tokenizers.implementations.BaseTokenizer):
    """Word-level tokenizer.

    Represents a simple word level tokenization using the internals of BERT's
    tokenizer.

    Based off the ``tokenizers`` `BertWordPieceTokenizer
    <https://github.com/huggingface/tokenizers/blob/704cf3fdd2f607ead58a561b892b510b49c301db/bindings/python/tokenizers/implementations/bert_wordpiece.py>`_.

    Parameters
    ----------
    vocab_file : str or pathlib.Path, optional
        Path to the vocabulary file.
        If not given, `word_id_map` must be given.
    word_id_map : dict, optional
        A dictionary mapping words to their IDs.
        If not given, `vocab_file` must be given.
    pad_token_id : int, optional
        The ID of the padding token.
        If not given, the padding token will not be added to the vocabulary.
    unk_token_id : int, optional
        The ID of the unknown token.
        If not given, the unknown token will not be added to the vocabulary.
    unk_token : str, default: "[UNK]"
        The unknown token.
    sep_token : str, default: "[SEP]"
        The separator token.
    cls_token : str, default: "[CLS]"
        The classifier token.
    pad_token : str, default: "[PAD]"
        The padding token.
    lowercase : bool, default: False
        Whether to lowercase the input text before tokenization or not.
    unicode_normalizer : {"nfc", "nfd", "nfkc", "nfkd"}, optional
        The unicode normalizer to use.

    """

    def __init__(
        self,
        vocab_file: Union[str, Path] = None,
        word_id_map: Optional[Dict[str, int]] = None,
        pad_token_id: Optional[int] = None,
        unk_token_id: Optional[int] = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        pad_token: str = "[PAD]",
        lowercase: bool = False,
        unicode_normalizer: Optional[str] = None,
    ) -> None:
        if vocab_file is None or not Path(vocab_file).is_file():
            assert word_id_map, "`vocab_file` or `word_id_map` must be specified."
            if pad_token_id:
                word_id_map[pad_token] = pad_token_id
            if unk_token_id:
                word_id_map[unk_token] = unk_token_id
            max_id = max(word_id_map.values())
            for idx, token in enumerate((unk_token, sep_token, cls_token, pad_token)):
                if token not in word_id_map:
                    word_id_map[token] = max_id + idx
            if vocab_file is None:
                # HuggingFace tokenizer expects a path to a `*.json` file to read the
                # vocab from. I think this is kind of a silly constraint, but for now
                # we write the vocab to a temporary file before initialization.
                vocab_file = tempfile.NamedTemporaryFile()
                vocab_file.write(json.dumps(word_id_map).encode())
                word_level = hf_tokenizers.models.WordLevel.from_file(
                    vocab_file.name, unk_token=str(unk_token)
                )
            else:  # vocab_file does not exist but word_id_map has been given
                vocab_file = Path(vocab_file).resolve()
                vocab_file.write_bytes(json.dumps(word_id_map).encode())
                word_level = hf_tokenizers.models.WordLevel.from_file(
                    str(vocab_file), unk_token=str(unk_token)
                )
        else:
            word_level = hf_tokenizers.models.WordLevel.from_file(
                str(vocab_file), unk_token=str(unk_token)
            )

        tokenizer = hf_tokenizers.Tokenizer(word_level)

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(sep_token)) is not None:
            tokenizer.add_special_tokens([str(sep_token)])
        if tokenizer.token_to_id(str(cls_token)) is not None:
            tokenizer.add_special_tokens([str(cls_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [
                hf_tokenizers.normalizers.unicode_normalizer_from_str(
                    unicode_normalizer
                )
            ]

        if lowercase:
            normalizers += [hf_tokenizers.normalizers.Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = hf_tokenizers.normalizers.Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = hf_tokenizers.pre_tokenizers.WhitespaceSplit()

        sep_token_id = tokenizer.token_to_id(str(sep_token))
        if sep_token_id is None:
            raise TypeError("sep_token not found in the vocabulary")
        cls_token_id = tokenizer.token_to_id(str(cls_token))
        if cls_token_id is None:
            raise TypeError("cls_token not found in the vocabulary")

        tokenizer.post_processor = hf_tokenizers.processors.BertProcessing(
            (str(sep_token), sep_token_id), (str(cls_token), cls_token_id)
        )

        parameters = {
            "model": "WordLevel",
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "pad_token": pad_token,
            "lowercase": lowercase,
            "unicode_normalizer": unicode_normalizer,
        }

        self.unk_token = unk_token
        self.pad_token = pad_token

        super().__init__(tokenizer, parameters)


class GloveTokenizer(WordLevelTokenizer):
    """A word-level tokenizer with GloVe n-dimensional vectors.

    Lowercased, since GloVe vectors are lowercased.

    Parameters
    ----------
    vocab_file : str or pathlib.Path, optional
        Path to the vocabulary file.
        If not given, `word_id_map` must be given.
    word_id_map : dict, optional
        A dictionary mapping words to their IDs.
        If not given, `vocab_file` must be given.
    pad_token_id : int, optional
        The ID of the padding token.
        If not given, the ID of the padding token will be the maximum ID in
        `word_id_map` plus 1.
    unk_token_id : int, optional
        The ID of the unknown token.
        If not given, the ID of the unknown token will be the maximum ID in
        `word_id_map` plus 2.
    max_length : int, default 256
        The maximum length of the tokenized sequence.
    ignore_punctuations : bool, default False
        Whether to ignore punctuations or not.
    words_to_ignore : List[str], optional
        A list of words to ignore.

    """

    def __init__(
        self,
        vocab_file: Union[str, Path] = None,
        word_id_map: Optional[Dict[str, int]] = None,
        pad_token_id: Optional[int] = None,
        unk_token_id: Optional[int] = None,
        max_length: int = 256,
        ignore_punctuations: bool = False,
        words_to_ignore: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(
            vocab_file=vocab_file,
            word_id_map=word_id_map,
            unk_token_id=unk_token_id,
            pad_token_id=pad_token_id,
            lowercase=True,
        )
        self.pad_token_id = self._tokenizer.token_to_id(self.pad_token)
        self.oov_token_id = self._tokenizer.token_to_id(self.unk_token)
        self.convert_id_to_word = self.id_to_token
        self.model_max_length = max_length
        self.ignore_punctuations = ignore_punctuations
        self.words_to_ignore = words_to_ignore
        # Set defaults.
        self.enable_padding(length=max_length, pad_id=self.pad_token_id)
        self.enable_truncation(max_length=max_length)

    def _process_text(self, text_input: Union[str, Sequence[str]]) -> List[str]:
        """
        A text input may be a single-input tuple (text,) or multi-input
        tuple (text, text, ...).

        In the single-input case, unroll the tuple. In the multi-input
        case, raise an error. Then, pre-tokenize the text input into a list of words.

        Parameters
        ----------
        text_input : str or Sequence[str]
            The text input to process.

        Returns
        -------
        List[str]
            The pre-tokenized text input.

        """
        if isinstance(text_input, (list, tuple)):
            if len(text_input) > 1:
                raise ValueError(
                    "Cannot use `GloveTokenizer` to encode multiple inputs"
                )
            text_input = text_input[0]
        text_input = tokenize(
            text_input,
            backend="nltk",
            words_to_ignore=self.words_to_ignore or [],
            ignore_punctuations=self.ignore_punctuations,
        )
        return text_input

    def encode(self, text: Union[str, Sequence[str]], **kwargs) -> List[int]:
        """Encode a text input into a list of word IDs.

        Parameters
        ----------
        text : str or Sequence[str]
            The text input to encode.

        Returns
        -------
        List[int]
            The encoded text input.

        """
        text = self._process_text(text)
        return super().encode(text, is_pretokenized=True, add_special_tokens=False).ids

    def batch_encode(self, input_text_list: Sequence[str], **kwargs) -> List[List[int]]:
        """The batch equivalent of :meth:`encode`.

        Parameters
        ----------
        input_text_list : Sequence[str]
            A list of text inputs to encode.

        Returns
        -------
        List[List[int]]
            The encoded text inputs.

        """
        input_text_list = list(map(self._process_text, input_text_list))
        encodings = self.encode_batch(
            input_text_list,
            is_pretokenized=True,
            add_special_tokens=False,
        )
        return [x.ids for x in encodings]

    def __call__(
        self,
        input_texts: Union[str, Sequence[str]],
        padding: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Union[torch.Tensor, List[int]]:
        """Method to encode a text input into a list of word IDs.

        Parameters
        ----------
        input_texts : str or Sequence[str]
            The text input to encode.
        padding : bool, default True
            Whether to pad the encoded text inputs.
        return_tensors : str, default "pt"
            Whether to return the encoded text inputs as PyTorch tensors.

        Returns
        -------
        torch.Tensor or List[int]
            The encoded text input.

        """
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        if padding:
            input_ids = self.batch_encode(input_texts)
            if return_tensors == "pt":
                return torch.tensor(input_ids)
        else:
            input_ids = [self.encode(text) for text in input_texts]
            if return_tensors == "pt":
                return [torch.tensor(x) for x in input_ids]
        return input_ids

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        """Convert a list of word IDs into a list of tokens (words).

        Parameters
        ----------
        ids : Sequence[int]
            The list of word IDs to convert.

        Returns
        -------
        List[str]
            The list of tokens (words).

        """
        return [self.convert_id_to_word(_id) for _id in ids]


def words_from_text(
    s: str, words_to_ignore: list = [], ignore_punctuations: bool = False
) -> list:
    """Split a string into a list of words.

    Works as a naive tokenizer.

    Parameters
    ----------
    s : str
        The string to split.
    words_to_ignore : list, default []
        A list of words to ignore.
    ignore_punctuations : bool, default False
        Whether to ignore punctuations.

    Returns
    -------
    list
        The list of words.

    """
    homos = "ɑЬϲԁе𝚏ɡհіϳ𝒌ⅼｍոорԛⲅѕ𝚝սѵԝ×уᴢ"
    exceptions = "'-_*@"
    filter_pattern = homos + "'\\-_\\*@"
    filter_pattern = f"[\\w{filter_pattern}]+"
    words = []
    for word in s.split():
        # Allow apostrophes, hyphens, underscores, asterisks and at signs as long as they don't begin the word.
        word = word.lstrip(exceptions)
        filt = [w.lstrip(exceptions) for w in re.findall(filter_pattern, word)]
        words.extend(filt)
    words = list(filter(lambda w: w not in words_to_ignore + [""], words))
    return words


def tokenize(
    s: str,
    backend: str = "nltk",
    words_to_ignore: list = [],
    ignore_punctuations: bool = False,
) -> List[str]:
    """Tokenize a string into a list of words.

    Parameters
    ----------
    s : str
        The string to tokenize.
    backend : {"nltk", "naive"}, default "nltk"
        The backend to use for tokenization.
    words_to_ignore : list, default []
        A list of words to ignore.
    ignore_punctuations : bool, default False
        Whether to ignore punctuations.

    Returns
    -------
    List[str]
        The list of tokens (words).

    """
    # _s = remove_space_before_punct(s)
    # TODO: deal with exceptions like `he 's`
    _s = s
    if backend.lower() == "nltk":
        if nltk_word_tokenize is None:
            warnings.warn("NLTK not installed. Using naive tokenizer.")
            words = words_from_text(_s)
        else:
            words = nltk_word_tokenize(_s)
    elif backend.lower() == "naive":
        words = words_from_text(_s)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    if ignore_punctuations:
        words = list(
            filter(lambda w: len(re.sub(f"[{punctuation}]+", "", w)) > 0, words)
        )
    words = list(filter(lambda w: w not in words_to_ignore + [""], words))
    return words
