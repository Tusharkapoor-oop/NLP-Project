"""Text preprocessing shared by training and inference.

The single most common production bug in NLP systems is *training/serving skew*:
the model was trained on one preprocessing pipeline and is served with another.
Everything that touches raw text therefore lives here, in one module, and both
the training scripts and the API import it.

Two pipelines are exposed because the two model families need different inputs:

* ``normalize_text`` / ``tokenize`` - whitespace tokens for TF-IDF and Word2Vec.
* ``transformer_input`` - raw text for a Hugging Face tokenizer, which performs
  its own sub-word splitting.

Resilience contract
-------------------
NLTK is a *training-time* dependency. The serving image deliberately does not
ship it (see ``requirements-serve.txt``), so every optional resource here has a
deterministic, documented fallback:

* stop-words  -> the bundled ``_FALLBACK_STOPWORDS`` list (a curated subset of
  NLTK's English list) instead of an empty set;
* lemmatizer  -> conservative plural stemming instead of no processing;
* NLTK data   -> discovered inside the repository (``.nltk_data``) even when the
  ``NLTK_DATA`` environment variable is unset.

The effect of a fallback is a small quality difference, never a crash and never
a silent change in *shape* (both paths return ``list[str]``).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

# --- regexes, compiled once at import time ---------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^<>]{1,500}>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# The repository ships its NLTK corpora next to the code so the project works
# on locked-down machines without touching the user profile directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_NLTK_DATA = _PROJECT_ROOT / ".nltk_data"

# Curated subset of NLTK's English stop-word list. Kept small on purpose: these
# are the function words that carry no topical signal. It is used only when the
# real corpus is unavailable, and the list is stable across releases.
_FALLBACK_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren as at be because
    been before being below between both but by can cannot could couldn did didn
    do does doesn doing don down during each few for from further had hadn has
    hasn have haven having he her here hers herself him himself his how i if in
    into is isn it its itself just let me more most mustn my myself no nor not
    now of off on once only or other ought our ours ourselves out over own same
    shan she should shouldn so some such than that the their theirs them
    themselves then there these they this those through to too under until up
    very was wasn we were weren what when where which while who whom why will
    with won would wouldn you your yours yourself yourselves
    """.split()
)


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


def _register_bundled_nltk_data() -> None:
    """Make the repository's ``.nltk_data`` discoverable without env vars.

    NLTK looks at ``NLTK_DATA`` and its default search paths only, so a checkout
    that keeps corpora inside the project would otherwise appear to have none.
    Registering the path is idempotent and cheap.
    """
    try:
        import nltk

        local = str(_BUNDLED_NLTK_DATA)
        if _BUNDLED_NLTK_DATA.exists() and local not in nltk.data.path:
            nltk.data.path.insert(0, local)
    except Exception:  # pragma: no cover - nltk is optional at serving time
        return


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    """NLTK's English stop-word list, loaded once.

    Falls back to a bundled list when the NLTK data is unavailable so that a
    missing data package degrades quality instead of silently disabling
    stop-word removal (which changes token distributions between train and
    serve).
    """
    _register_bundled_nltk_data()
    try:
        from nltk.corpus import stopwords

        words = frozenset(stopwords.words("english"))
        if words:
            return words
    except Exception:  # pragma: no cover - depends on local NLTK data
        pass
    return _FALLBACK_STOPWORDS


class _PluralLemmatizer:
    """Conservative suffix stemmer used when NLTK's WordNet is unavailable.

    Only plurals and possessive 's are folded. It is intentionally timid: an
    aggressive stemmer (Porter, Snowball) changes far more tokens than WordNet
    does, which would *create* the train/serve skew this module exists to
    prevent. Plural folding is the highest-value, lowest-risk subset.
    """

    __slots__ = ()

    def lemmatize(self, token: str) -> str:
        if len(token) <= 3 or not token.isalpha():
            return token
        if token.endswith("'s"):
            return token[:-2]
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith(("sses", "shes", "ches", "xes", "zes")):
            return token[:-2]
        if token.endswith("s") and not token.endswith(("ss", "us", "is", "ics")):
            return token[:-1]
        return token


@lru_cache(maxsize=1)
def _lemmatizer():
    """WordNet lemmatizer, or a conservative plural stemmer as fallback.

    Serving must never hard-fail because a training-time data package is
    missing; degrading keeps the service alive and the effect on accuracy is
    small (the vectors were trained with lemmas, so most tokens still match).
    """
    _register_bundled_nltk_data()
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")  # forces the WordNet corpus to load
        return lemmatizer
    except Exception:  # pragma: no cover - depends on local NLTK data
        return _PluralLemmatizer()


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
