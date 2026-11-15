"""Model tier implementations.

Each tier wraps one trained artifact and exposes the same two methods:
``predict_proba(texts) -> ndarray[n, n_classes]`` and a ``ready`` flag.
The API layer never needs to know which framework is underneath.
"""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .text import normalize_text, tokenize, transformer_input


class TierModel(ABC):
    """Common interface implemented by every tier."""

    name: str

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @property
    @abstractmethod
    def ready(self) -> bool: ...

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...

    def predict(self, texts: list[str]) -> tuple[list[int], np.ndarray]:
        probabilities = self.predict_proba(texts)
        indices = [int(i) for i in probabilities.argmax(axis=1)]
        return indices, probabilities

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


# ---------------------------------------------------------------------------
class BronzeModel(TierModel):
    """TF-IDF + Logistic Regression (scikit-learn)."""

    name = "bronze"

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        self._pipeline = None
        self._labels: list[str] = []
        self._load()

    def _load(self) -> None:
        path = self.directory / "model.joblib"
        if not path.exists():
            return
        import joblib

        self._pipeline = joblib.load(path)
        metrics = self.directory / "metrics.json"
        self._labels = (
            json.loads(metrics.read_text(encoding="utf-8")).get("labels")
            if metrics.exists()
            else None
        ) or ["World", "Sports", "Business", "Sci/Tech"]

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("bronze model is not loaded")
        cleaned = [normalize_text(t) for t in texts]
        with self._lock:
            return self._pipeline.predict_proba(cleaned)


# ---------------------------------------------------------------------------
class SilverModel(TierModel):
    """Word2Vec embeddings + BiLSTM (PyTorch)."""

    name = "silver"

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        self._model = None
        self._vocab: dict[str, int] = {}
        self._max_len = 64
        self._labels: list[str] = []
        self._torch = None
        self._load()

    def _load(self) -> None:
        weights = self.directory / "model.pt"
        vocab_path = self.directory / "vocab.json"
        if not (weights.exists() and vocab_path.exists()):
            return

        import torch
        import torch.nn as nn

        payload = json.loads(vocab_path.read_text(encoding="utf-8"))
        vocabulary = payload["vocabulary"]
        self._vocab = {w: i for i, w in enumerate(vocabulary)}
        self._max_len = int(payload.get("max_len", 64))
        self._labels = payload.get("labels", ["World", "Sports", "Business", "Sci/Tech"])
        self._torch = torch

        torch.set_num_threads(max(1, min(4, torch.get_num_threads() or 4)))

        class _Net(nn.Module):
            def __init__(self, n_words: int, dim: int = 100, hidden: int = 128, n_classes: int = 4):
                super().__init__()
                self.embedding = nn.Embedding(n_words, dim, padding_idx=0)
                self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
                self.dropout = nn.Dropout(0.4)
                self.classifier = nn.Linear(hidden * 2, n_classes)

            def forward(self, input_ids):
                embedded = self.dropout(self.embedding(input_ids))
                outputs, _ = self.lstm(embedded)
                mask = (input_ids != 0).unsqueeze(-1).float()
                pooled = (outputs * mask).sum(1) / mask.sum(1).clamp(min=1)
                return self.classifier(self.dropout(pooled))

        # The architecture must match training exactly; build then load weights.
        net = _Net(len(vocabulary))
        state = torch.load(weights, map_location="cpu")
        net.load_state_dict(state)
        net.eval()
        self._model = net

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _encode(self, text: str) -> list[int]:
        tokens = tokenize(text)
        ids = [self._vocab.get(t, 1) for t in tokens][: self._max_len]
        return ids or [1]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("silver model is not loaded")
        torch = self._torch
        sequences = [torch.tensor(self._encode(t)) for t in texts]
        batch = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
        with self._lock, torch.no_grad():
            logits = self._model(batch)
            return torch.softmax(logits, dim=-1).numpy()


# ---------------------------------------------------------------------------
class GoldModel(TierModel):
    """Fine-tuned DistilBERT (transformers)."""

    name = "gold"

    def __init__(self, directory: Path, max_length: int = 96) -> None:
        super().__init__()
        self.directory = directory
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._labels: list[str] = []
        self._batch_size = 16
        self._load()

    def _load(self) -> None:
        config_path = self.directory / "config.json"
        if not config_path.exists():
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(max(1, min(4, torch.get_num_threads() or 4)))
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.directory))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(self.directory))
        self._model.eval()
        self._torch = torch  # kept for the explainability path
        id2label = getattr(self._model.config, "id2label", {}) or {}
        self._labels = [id2label[i] for i in sorted(id2label)] if id2label else []

    @property
    def ready(self) -> bool:
        return self._model is not None

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("gold model is not loaded")
        import torch

        cleaned = [transformer_input(t) for t in texts]
        with self._lock, torch.no_grad():
            batch = self._tokenizer(
                cleaned,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            logits = self._model(**batch).logits
            return torch.softmax(logits, dim=-1).numpy()


# ---------------------------------------------------------------------------
TIER_CLASSES: dict[str, type[TierModel]] = {
    "bronze": BronzeModel,
    "silver": SilverModel,
    "gold": GoldModel,
}

TIER_DISPLAY = {
    "bronze": "Bronze - TF-IDF + Logistic Regression",
    "silver": "Silver - Word2Vec + BiLSTM",
    "gold": "Gold - Fine-tuned DistilBERT",
}

TIER_DESCRIPTION = {
    "bronze": (
        "Cheapest and fastest. Bag-of-words model with no neural network. "
        "Serves high-volume, low-stakes traffic."
    ),
    "silver": (
        "Balanced. Learns its own embeddings from the corpus, then uses a "
        "bidirectional LSTM. Good accuracy without a GPU."
    ),
    "gold": (
        "Highest quality. Pre-trained DistilBERT fine-tuned on this task. "
        "Best accuracy, highest cost per request."
    ),
}


def load_tier(tier: str, artifacts_dir: Path) -> TierModel:
    """Instantiate one tier. Returns an unready instance if artifacts are missing."""
    cls = TIER_CLASSES[tier]
    return cls(artifacts_dir / tier)  # type: ignore[call-arg]


def timed(fn):
    """Small helper used by the API for latency accounting."""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        return result, (time.perf_counter() - start) * 1000

    return wrapper
