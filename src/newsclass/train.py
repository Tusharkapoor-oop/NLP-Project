"""Train the three model tiers and register them for serving.

This module is the single source of truth for *how* each tier is produced, so a
reviewer can re-run it end-to-end and get the same artifacts.

Tiers (all evaluated on the same held-out test set so they are comparable):

* **bronze** - TF-IDF + Logistic Regression. Cheap, fast, no PyTorch.
* **silver** - Word2Vec (CBOW) embeddings + BiLSTM. Modern classic, CPU training.
* **gold**   - DistilBERT fine-tuned. Best accuracy, heaviest to train.

Run:  python -m newsclass.train --tier all
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import env_artifacts_dir
from .text import normalize_text, tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED = 42
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]  # AG News, id -> name
DATASET_NAME = "SetFit/ag_news"
TRAIN_ROWS = 20_000
MAX_LEN = 64
MAX_VOCAB = 20_000
EMBEDDING_DIM = 100


@dataclass
class TierMetrics:
    """Everything we need to justify a tier's cost and quality in review."""

    tier: str
    model_name: str
    model_version: str
    accuracy: float
    macro_f1: float
    train_seconds: float
    inference_ms_per_100: float
    size_mb: float
    n_train: int
    n_test: int
    trained_at: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _seed_everything(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def load_split(rows: int = TRAIN_ROWS, *, train_rows: int | None = None):
    """Load AG News and carve a **canonical** stratified train/val/test split.

    The partition is computed over a fixed universe of ``rows`` examples and never
    depends on how much data a tier chooses to train on. That is a correctness
    requirement, not tidiness:

    A previous version of this function partitioned whatever range it was asked
    for, so ``load_split(rows=6000)`` and ``load_split(rows=20000)`` produced
    *different* test sets from overlapping source rows. The gold tier trained on
    the first partition and was then scored on the second, which meant 464 of its
    own training documents (23.2% of the test set) reappeared at evaluation time.
    Gold scored 95.7% on those memorised documents and 90.2% on the rest — an
    inflated headline number caused entirely by comparing two different splits.

    Now ``rows`` sets the universe, ``train_rows`` optionally subsamples **only**
    the training portion, and the validation/test partitions are byte-identical
    for every tier. Every tier is therefore scored on the same held-out documents,
    which is what makes the tier table a fair comparison.

    Set ``train_rows`` (not ``rows``) to train on a subset — for example the
    CPU-bound gold tier.
    """
    # Point Hugging Face at the project's own cache *before* importing the
    # library. Every script does this, but ``load_split`` is also called from the
    # library and from tests, where a user-level cache may hold a half-written
    # ``*.incomplete`` download that makes ``load_dataset`` stall on the network.
    # ``setdefault`` keeps any cache the caller already chose.
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "datasets"))

    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    raw = load_dataset(DATASET_NAME)
    frame = raw["train"].select(range(min(rows, len(raw["train"])))).to_pandas()
    frame = frame.rename(columns={"label": "label"})
    frame["topic"] = frame["label"].map(dict(enumerate(LABEL_NAMES)))

    train_df, rest = train_test_split(
        frame, test_size=0.2, random_state=SEED, stratify=frame["label"]
    )
    val_df, test_df = train_test_split(
        rest, test_size=0.5, random_state=SEED, stratify=rest["label"]
    )

    # Subsample only the training portion. Stratified so class balance survives,
    # and seeded so the subset is reproducible. Validation and test are untouched:
    # shrinking the training set must never move the held-out documents.
    if train_rows is not None and 0 < train_rows < len(train_df):
        train_df, _discarded = train_test_split(
            train_df,
            train_size=train_rows,
            random_state=SEED,
            stratify=train_df["label"],
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _evaluate_predictions(y_true, y_pred) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro")),
    )


# ---------------------------------------------------------------------------
# BRONZE: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------
def train_bronze(out_dir: Path) -> TierMetrics:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    started = time.time()
    train_df, _val_df, test_df = load_split()

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 1), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    # CRITICAL: train on exactly the same representation the API serves.
    # Training on raw text but serving normalised text is training/serving skew.
    train_texts = [normalize_text(t) for t in train_df["text"]]
    test_texts = [normalize_text(t) for t in test_df["text"]]
    pipeline.fit(train_texts, train_df["label"])

    predicted = pipeline.predict(test_texts)
    accuracy, macro_f1 = _evaluate_predictions(test_df["label"], predicted)

    # Measure realistic single-item latency.
    sample = test_texts[:100]
    t0 = time.perf_counter()
    pipeline.predict(sample)
    latency_ms = (time.perf_counter() - t0) * 1000

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    size_mb = model_path.stat().st_size / 1e6
    return TierMetrics(
        tier="bronze",
        model_name="tfidf+logreg",
        model_version="1.0.0",
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        train_seconds=round(time.time() - started, 1),
        inference_ms_per_100=round(latency_ms, 2),
        size_mb=round(size_mb, 2),
        n_train=len(train_df),
        n_test=len(test_df),
        trained_at=_now_iso(),
        notes="No PyTorch. Cheapest tier; strong baseline on short news text.",
    )


# ---------------------------------------------------------------------------
# SILVER: Word2Vec (CBOW) + BiLSTM
# ---------------------------------------------------------------------------
def train_silver(out_dir: Path) -> TierMetrics:
    from collections import Counter

    import torch
    import torch.nn as nn
    from gensim.models import Word2Vec
    from torch.utils.data import DataLoader, Dataset

    _seed_everything()
    torch.set_num_threads(max(1, min(4, (torch.get_num_threads() or 4))))
    started = time.time()

    train_df, val_df, test_df = load_split()
    for frame in (train_df, val_df, test_df):
        frame["tokens"] = frame["text"].map(
            lambda t: tokenize(t, use_stopwords=True, use_lemma=True)
        )
    train_df = train_df[train_df["tokens"].map(len) > 0].reset_index(drop=True)

    counts = Counter(token for tokens in train_df["tokens"] for token in tokens)
    vocabulary = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(MAX_VOCAB - 2)]
    word_to_id = {w: i for i, w in enumerate(vocabulary)}

    w2v = Word2Vec(
        sentences=train_df["tokens"].tolist(),
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=2,
        sg=0,  # CBOW: better neighbours and faster on this corpus
        workers=4,
        seed=SEED,
        epochs=20,
    )

    embedding_matrix = np.random.normal(0, 0.1, (len(vocabulary), EMBEDDING_DIM)).astype("float32")
    embedding_matrix[0] = 0.0
    for word, index in word_to_id.items():
        if word in w2v.wv:
            embedding_matrix[index] = w2v.wv[word]

    def to_ids(tokens: list[str]) -> list[int]:
        ids = [word_to_id.get(t, 1) for t in tokens][:MAX_LEN]
        return ids or [1]

    for frame in (train_df, val_df, test_df):
        frame["ids"] = frame["tokens"].map(to_ids)

    class _Dataset(Dataset):
        def __init__(self, frame):
            self.x = frame["ids"].tolist()
            self.y = frame["label"].tolist()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return torch.tensor(self.x[i]), torch.tensor(self.y[i])

    def _collate(batch):
        seqs, labels = zip(*batch, strict=False)
        return (
            nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0),
            torch.stack(labels),
        )

    class BiLSTMClassifier(nn.Module):
        def __init__(self, matrix, hidden=128, n_classes=4, dropout=0.4):
            super().__init__()
            n_words, dim = matrix.shape
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(matrix), freeze=False, padding_idx=0
            )
            self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden * 2, n_classes)

        def forward(self, input_ids):
            embedded = self.dropout(self.embedding(input_ids))
            outputs, _ = self.lstm(embedded)
            mask = (input_ids != 0).unsqueeze(-1).float()
            pooled = (outputs * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.classifier(self.dropout(pooled))

    train_loader = DataLoader(_Dataset(train_df), batch_size=64, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(_Dataset(val_df), batch_size=128, collate_fn=_collate)
    test_loader = DataLoader(_Dataset(test_df), batch_size=128, collate_fn=_collate)

    model = BiLSTMClassifier(embedding_matrix, n_classes=len(LABEL_NAMES))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def run_epoch(loader, opt=None):
        training = opt is not None
        model.train() if training else model.eval()
        preds, gold = [], []
        with torch.enable_grad() if training else torch.no_grad():
            for x, y in loader:
                logits = model(x)
                loss = criterion(logits, y)
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    opt.zero_grad()
                preds.extend(logits.argmax(1).tolist())
                gold.extend(y.tolist())
        return _evaluate_predictions(gold, preds)

    best_f1, best_state = 0.0, None
    for _epoch in range(6):
        run_epoch(train_loader, optimizer)
        _, val_f1 = run_epoch(val_loader)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_f1 = run_epoch(test_loader)
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.extend(model(x).argmax(1).tolist())
            gold.extend(y.tolist())
    accuracy, macro_f1 = _evaluate_predictions(gold, preds)

    # latency over 100 items
    with torch.no_grad():
        smoke = torch.randint(1, min(1000, len(vocabulary)), (100, MAX_LEN))
        t0 = time.perf_counter()
        model(smoke)
        latency_ms = (time.perf_counter() - t0) * 1000

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "model.pt"
    torch.save(model.state_dict(), weights_path)
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"vocabulary": vocabulary, "max_len": MAX_LEN, "labels": LABEL_NAMES},
            fh,
        )

    size_mb = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file()) / 1e6
    return TierMetrics(
        tier="silver",
        model_name="word2vec-cbow+bilstm",
        model_version="1.0.0",
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        train_seconds=round(time.time() - started, 1),
        inference_ms_per_100=round(latency_ms, 2),
        size_mb=round(size_mb, 2),
        n_train=len(train_df),
        n_test=len(test_df),
        trained_at=_now_iso(),
        notes="Own embeddings trained on the training split only (no leakage).",
    )


# ---------------------------------------------------------------------------
# GOLD: fine-tuned DistilBERT
# ---------------------------------------------------------------------------
def train_gold(out_dir: Path, epochs: int = 2, rows: int = 6000) -> TierMetrics:
    """Fine-tune DistilBERT on AG News.

    Two practical notes:

    * ``rows`` defaults to 6,000 raw examples (4,800 after the split). Training
      on the full 16,000 takes multiple hours on a 4-thread CPU, which is not a
      reasonable default. 6,000 is enough for DistilBERT to clearly outperform
      the classical tiers while finishing in a workable time.
    * Defaults to 2 epochs: validation macro-F1 plateaus by then on this data.
    * ``max_length=64``: AG News leads average ~55 sub-word tokens, so 64 keeps
      almost every article whole and runs roughly twice as fast as 96 on CPU.
    """
    import torch
    from datasets import Dataset as HFDataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    _seed_everything()
    set_seed(SEED)
    started = time.time()

    # ``train_rows`` subsamples only the training portion, so the validation and
    # test partitions stay identical to every other tier's. Passing ``rows`` here
    # would move the test set and silently put gold's own training documents into
    # its evaluation — the leakage this function is now written to prevent.
    train_df, val_df, test_df = load_split(train_rows=rows)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # Measured on AG News: articles average ~55 sub-word tokens, so 64 keeps
    # essentially every headline whole while halving CPU time versus 96
    # (attention cost grows with sequence length).
    max_length = 64

    def _to_hf(frame):
        ds = HFDataset.from_dict({"text": frame["text"].tolist(), "label": frame["label"].tolist()})
        return ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
            batched=True,
        )

    train_ds, val_ds, test_ds = _to_hf(train_df), _to_hf(val_df), _to_hf(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABEL_NAMES),
        id2label=dict(enumerate(LABEL_NAMES)),
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        warmup_steps=50,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    trainer.train()

    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    accuracy = float(accuracy_score(predictions.label_ids, preds))
    macro_f1 = float(f1_score(predictions.label_ids, preds, average="macro"))

    model.eval()
    batch = tokenizer(
        [test_df["text"].iloc[0]] * 100,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        t0 = time.perf_counter()
        model(**batch)
        latency_ms = (time.perf_counter() - t0) * 1000

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Record the exact truncation window used during training.
    #
    # This is not bookkeeping: the serving tier must truncate at the same length
    # it was trained on. Serving at a different window is training/serving skew —
    # the model sees sequences of a length it never saw during fine-tuning, and
    # the degradation is silent because nothing errors. Writing the value here
    # makes the two sides read from one source of truth instead of each holding
    # its own default, which is how they drifted apart before.
    (out_dir / "serving.json").write_text(
        json.dumps(
            {
                "max_length": max_length,
                "labels": LABEL_NAMES,
                "trained_at": _now_iso(),
                "epochs": epochs,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Remove the Trainer scratch directory; only the final model is served.
    scratch = out_dir / "_trainer"
    if scratch.exists():
        for child in sorted(scratch.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()
        scratch.rmdir()

    size_mb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6
    return TierMetrics(
        tier="gold",
        model_name="distilbert-base-uncased",
        model_version="1.0.0",
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        train_seconds=round(time.time() - started, 1),
        inference_ms_per_100=round(latency_ms, 2),
        size_mb=round(size_mb, 2),
        n_train=len(train_df),
        n_test=len(test_df),
        trained_at=_now_iso(),
        notes=f"Fine-tuned {epochs} epochs, max_length={max_length}.",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
BUILDERS = {"bronze": train_bronze, "silver": train_silver, "gold": train_gold}


def write_registry(artifacts_dir: Path) -> dict[str, Any]:
    """Merge per-tier metadata files into one registry the API reads."""
    entries: dict[str, Any] = {}
    for tier in ("bronze", "silver", "gold"):
        meta_path = artifacts_dir / tier / "metrics.json"
        if meta_path.exists():
            entries[tier] = json.loads(meta_path.read_text(encoding="utf-8"))

    registry = {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "labels": LABEL_NAMES,
        "dataset": {"name": DATASET_NAME, "train_rows": TRAIN_ROWS},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "tiers": entries,
    }
    (artifacts_dir / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Train news classification tiers")
    parser.add_argument("--tier", default="all", choices=["all", "bronze", "silver", "gold"])
    parser.add_argument("--artifacts-dir", default=str(env_artifacts_dir()))
    parser.add_argument("--gold-epochs", type=int, default=2)
    parser.add_argument(
        "--gold-rows",
        type=int,
        default=6000,
        help="raw examples used to fine-tune gold (CPU-friendly default)",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    tiers = ["bronze", "silver", "gold"] if args.tier == "all" else [args.tier]
    for tier in tiers:
        print(f"\n=== training {tier} ===", flush=True)
        out_dir = artifacts_dir / tier
        builder = BUILDERS[tier]
        if tier == "gold":
            metrics = builder(out_dir, epochs=args.gold_epochs, rows=args.gold_rows)
        else:
            metrics = builder(out_dir)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
        )
        print(
            f"  accuracy={metrics.accuracy} macro_f1={metrics.macro_f1} "
            f"({metrics.train_seconds}s, {metrics.size_mb} MB)",
            flush=True,
        )

    registry = write_registry(artifacts_dir)
    print(f"\nregistry written: {artifacts_dir / 'registry.json'}")
    print(json.dumps({k: v.get("accuracy") for k, v in registry["tiers"].items()}, indent=2))


if __name__ == "__main__":
    main()
