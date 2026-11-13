"""Text preprocessing shared by training and inference.

The single most common production bug in NLP systems is *training/serving skew*:
the model was trained on one preprocessing pipeline and is served with another.
Everything that touches raw text therefore lives here, in one module, and both
the training scripts and the API import it.

Two pipelines are exposed because the two model families need different inputs:

* ``normalize_text`` / ``tokenize`` - whitespace tokens for TF-IDF and Word2Vec.
* ``transformer_input`` - raw text for a Hugging Face tokenizer, which performs
  its own sub-word splitting.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# --- regexes, compiled once at import time ---------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^<>]{1,500}>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, *, keep_html: bool = False) -> str:
    """Lower-case, strip URLs/HTML/symbols and collapse whitespace.

    Numbers are removed on purpose: for topic classification they add noise
    rather than signal, and keeping them made the vocabulary noticeably larger
    without improving validation accuracy.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")

    # Normalise unicode so accented characters and ligatures behave consistently.
    text = unicodedata.normalize("NFKC", text).lower()
    text = _URL_RE.sub(" ", text)
    if not keep_html:
        text = _HTML_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    """NLTK's English stop-word list, loaded once.

    Falls back to an empty set when the NLTK data is unavailable so that a
    missing data package degrades quality instead of crashing the service.
    """
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("english"))
    except Exception:  # pragma: no cover - depends on local NLTK data
        return frozenset()


@lru_cache(maxsize=1)
def _lemmatizer():
    """WordNet lemmatizer, or ``None`` when NLTK data is unavailable.

    Serving must never hard-fail because a training-time data package is
    missing; degrading to no lemmatization keeps the service alive and the
    effect on accuracy is small (the vectors were trained with lemmas, so most
    tokens still match).
    """
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")  # forces the WordNet corpus to load
        return lemmatizer
    except Exception:  # pragma: no cover - depends on local NLTK data
        return None


def tokenize(text: str, *, use_stopwords: bool = True, use_lemma: bool = True) -> list[str]:
    """Split already-normalised text into lemmatised content words."""
    stops = _stopwords() if use_stopwords else frozenset()
    lemmatizer = _lemmatizer() if use_lemma else None
    tokens = []
    for token in normalize_text(text).split():
        if token in stops or len(token) <= 2:
            continue
        tokens.append(lemmatizer.lemmatize(token) if lemmatizer is not None else token)
    return tokens


def transformer_input(text: str, max_chars: int = 20_000) -> str:
    """Prepare raw text for a Hugging Face tokenizer.

    Transformer tokenizers expect natural casing and punctuation, so we do *not*
    lowercase or strip symbols here. We only normalise unicode, drop HTML and
    hard-cap the length so a pathological document cannot exhaust memory.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:max_chars]


def warm_up() -> None:
    """Touch the lazy resources so the first real request is not slow."""
    _stopwords()
    _lemmatizer()
