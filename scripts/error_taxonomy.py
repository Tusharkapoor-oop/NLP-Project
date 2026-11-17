"""Error taxonomy and subgroup slices: *where* a model fails, not just how often.

Why this exists
---------------
A single accuracy number buries every interesting fact about a model. Two models
at 0.90 can fail in completely different ways, and a reviewer's most useful
question is not "how good is it" but **"what does it get wrong, and does that
matter?"**

This script answers that in two ways:

1. **A failure taxonomy.** Every misclassification is assigned to a category
   using signals available without human labelling — so the categories are
   reproducible rather than a hand-picked anecdote:

   * ``long_document`` — the input is longer than the model can read. For the
     Transformer this is truncation (``max_length``); for the BiLSTM it is the
     ``MAX_LEN`` window. This is the failure mode this project already documents,
     so it is the first thing to quantify.
   * ``confident_error`` — wrong, but with high confidence. The dangerous class
     operationally, because a threshold on confidence would not catch it.
   * ``near_miss_boundary`` — wrong because two topics genuinely overlap
     (Business vs Sci/Tech is the classic pair). Detected as: correct label is
     the model's second choice.
   * ``rare_token_overlap`` — most of the input's content words never appear in
     the training vocabulary, i.e. the model had little to work with.
   * ``other`` — wrong without an obvious structural cause.

2. **Subgroup slices.** The same model, scored on disjoint subsets: by document
   length tercile, by class, and by whether the text is headline-shaped or
   article-shaped. A model that is excellent on short headlines and poor on long
   articles is a different product from one that is uniformly mediocre — and only
   the slice view shows which you have.

Nothing here replaces a proper fairness audit on demographic data (this corpus
has none). It is a *behavioural* slice analysis, and it is described as such.

Usage
-----
    python scripts/error_taxonomy.py                     # all tiers, 1000 docs
    python scripts/error_taxonomy.py --tier gold --examples 8
    python scripts/error_taxonomy.py --json-out taxonomy.json

Exit codes: 0 always (this is a diagnostic); 2 if nothing could be evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np  # noqa: E402

from newsclass.config import env_artifacts_dir  # noqa: E402
from newsclass.registry import ModelRegistry  # noqa: E402
from newsclass.text import tokenize  # noqa: E402
from newsclass.train import LABEL_NAMES, load_split  # noqa: E402

# How many characters map to one model token, very roughly. Only used to decide
# whether a document is *likely* truncated, so an approximation is acceptable;
# the tokenizer itself is used when available (see token_length()).
APPROX_CHARS_PER_TOKEN = 4.0
CONFIDENT_ERROR_THRESHOLD = 0.80


def token_length(text: str, tokenizer=None) -> int:
    """Sub-word length if a tokenizer is available, else a character estimate."""
    if tokenizer is not None:
        try:
            return len(tokenizer(text, truncation=False)["input_ids"])
        except Exception:  # noqa: BLE001 - fall back rather than crash a report
            pass
    return int(len(text) / APPROX_CHARS_PER_TOKEN)


def vocab_coverage(text: str, vocabulary: set[str]) -> float:
    """Fraction of the text's content words that the model's vocab contains."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token in vocabulary) / len(tokens)


def load_vocabulary(registry: ModelRegistry, tier: str) -> set[str]:
    """The silver tier's vocabulary is on disk; other tiers have no explicit one."""
    if tier != "silver":
        return set()
    path = registry.artifacts_dir / "silver" / "vocab.json"
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return set(payload.get("vocabulary", []))
    except (json.JSONDecodeError, OSError):
        return set()


def classify_failure(
    *,
    text: str,
    correct_label: str,
    predicted_label: str,
    confidence: float,
    second_choice: str | None,
    tokens: int,
    max_tokens: int,
    coverage: float,
) -> str:
    """Assign one wrong prediction to a category.

    Order matters: the categories are checked from most specific to least, so a
    long document that is also a confident error is reported as ``long_document``
    (the actionable structural cause) rather than the vaguer label.
    """
    if tokens > max_tokens:
        return "long_document"
    if second_choice == correct_label:
        return "near_miss_boundary"
    if confidence >= CONFIDENT_ERROR_THRESHOLD:
        return "confident_error"
    if coverage and coverage < 0.6:
        return "rare_token_overlap"
    return "other"


def thirdiles(values: list[float]) -> tuple[float, float]:
    """Cut points for splitting a distribution into three equal parts."""
    if not values:
        return 0.0, 0.0
    return (
        float(np.percentile(values, 33.33)),
        float(np.percentile(values, 66.67)),
    )


def evaluate_tier(
    registry: ModelRegistry,
    tier: str,
    frame,
    *,
    max_tokens: int,
    examples: int,
) -> dict:
    model = registry.get(tier)
    texts = frame["text"].tolist()
    gold = frame["label"].tolist()
    probabilities = model.predict_proba(texts)

    tokenizer = getattr(model, "_tokenizer", None)
    vocabulary = load_vocabulary(registry, tier)

    lengths = [token_length(text, tokenizer) for text in texts]
    low_cut, high_cut = thirdiles([float(value) for value in lengths])

    predicted = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    categories: Counter[str] = Counter()
    examples_by_category: dict[str, list[dict]] = defaultdict(list)

    for index, (text, truth, guess) in enumerate(zip(texts, gold, predicted, strict=True)):
        if int(guess) == int(truth):
            continue
        order = np.argsort(probabilities[index])[::-1]
        second = LABEL_NAMES[int(order[1])] if len(order) > 1 else None
        category = classify_failure(
            text=text,
            correct_label=LABEL_NAMES[int(truth)],
            predicted_label=LABEL_NAMES[int(guess)],
            confidence=float(confidences[index]),
            second_choice=second,
            tokens=lengths[index],
            max_tokens=max_tokens,
            coverage=vocab_coverage(text, vocabulary) if vocabulary else 0.0,
        )
        categories[category] += 1
        if len(examples_by_category[category]) < examples:
            examples_by_category[category].append(
                {
                    "text": text[:280],
                    "truth": LABEL_NAMES[int(truth)],
                    "predicted": LABEL_NAMES[int(guess)],
                    "confidence": round(float(confidences[index]), 4),
                    "tokens": lengths[index],
                }
            )

    # -- slices -----------------------------------------------------------
    def slice_stats(indices: list[int], name: str) -> dict:
        if not indices:
            return {"name": name, "documents": 0}
        truths = np.asarray([gold[i] for i in indices])
        guesses = np.asarray([predicted[i] for i in indices])
        return {
            "name": name,
            "documents": len(indices),
            "accuracy": round(float((truths == guesses).mean()), 4),
            "mean_confidence": round(float(np.mean([confidences[i] for i in indices])), 4),
        }

    length_slices = []
    for name, predicate in (
        (f"short (<={low_cut:.0f} tokens)", lambda i: lengths[i] <= low_cut),
        ("medium", lambda i: low_cut < lengths[i] <= high_cut),
        (f"long (>{high_cut:.0f} tokens)", lambda i: lengths[i] > high_cut),
    ):
        length_slices.append(slice_stats([i for i in range(len(texts)) if predicate(i)], name))

    class_slices = [
        slice_stats(
            [i for i in range(len(texts)) if gold[i] == label], f"class: {LABEL_NAMES[label]}"
        )
        for label in range(len(LABEL_NAMES))
    ]

    # Headline-shaped vs article-shaped. AG News leads are typically one or two
    # sentences; longer inputs are closer to a full article opening.
    sentence_slices = [
        slice_stats(
            [i for i in range(len(texts)) if texts[i].count(".") < 3],
            "headline-shaped (<3 sentences)",
        ),
        slice_stats(
            [i for i in range(len(texts)) if texts[i].count(".") >= 3],
            "article-shaped (>=3 sentences)",
        ),
    ]

    total_wrong = int((predicted != np.asarray(gold)).sum())
    return {
        "tier": tier,
        "documents": len(texts),
        "correct": len(texts) - total_wrong,
        "wrong": total_wrong,
        "accuracy": round(1 - total_wrong / max(1, len(texts)), 4),
        "max_tokens_read": max_tokens,
        "token_length": {
            "mean": round(float(np.mean(lengths)), 1),
            "p50": int(np.median(lengths)),
            "p95": int(np.percentile(lengths, 95)),
            "max": int(max(lengths)),
        },
        "failure_categories": [
            {
                "category": category,
                "count": count,
                "share_of_errors": round(count / max(1, total_wrong), 4),
                "examples": examples_by_category.get(category, []),
            }
            for category, count in categories.most_common()
        ],
        "slices": {
            "by_length": length_slices,
            "by_class": class_slices,
            "by_shape": sentence_slices,
        },
    }


def print_tier(result: dict) -> None:
    print(f"\n{'=' * 78}")
    print(
        f"{result['tier'].upper()}  —  {result['correct']}/{result['documents']} correct "
        f"({result['accuracy']:.4f}), {result['wrong']} errors analysed"
    )
    print("=" * 78)
    tl = result["token_length"]
    reads = (
        f"the model reads at most {result['max_tokens_read']} tokens"
        if result.get("truncates_input", True)
        else "the model reads the whole document (no token window)"
    )
    print(
        f"  tokens: mean {tl['mean']:.0f}, p50 {tl['p50']}, p95 {tl['p95']}, "
        f"max {tl['max']}  |  {reads}"
    )

    print("\n  FAILURE TAXONOMY (share of that tier's errors)")
    if not result["failure_categories"]:
        print("    no errors — nothing to categorise")
    for entry in result["failure_categories"]:
        bar = "#" * max(1, int(entry["share_of_errors"] * 40))
        print(
            f"    {entry['category']:<22} {entry['count']:>4}  "
            f"{entry['share_of_errors']:>6.1%}  {bar}"
        )
        for example in entry["examples"][:2]:
            print(
                f"        truth={example['truth']:<9} got={example['predicted']:<9} "
                f"conf={example['confidence']:.2f}  {example['text'][:58]}"
            )

    print("\n  SLICES — the same model, different subsets")
    for group, entries in result["slices"].items():
        print(f"    {group}:")
        for entry in entries:
            if not entry.get("documents"):
                continue
            print(
                f"      {entry['name']:<30} n={entry['documents']:>4}  "
                f"acc={entry['accuracy']:.4f}  conf={entry['mean_confidence']:.3f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Failure taxonomy and subgroup slices")
    parser.add_argument("--tier", default="all", choices=["all", "bronze", "silver", "gold"])
    parser.add_argument("--sample", type=int, default=1000, help="0 = the whole test split")
    parser.add_argument("--examples", type=int, default=3, help="examples kept per category")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="tokens the tier can read; inferred from the model where possible",
    )
    parser.add_argument("--artifacts-dir", default=str(env_artifacts_dir()))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    registry = ModelRegistry(artifacts)
    available = registry.available_tiers()
    if not available:
        print(f"No trained tiers found in {artifacts}", file=sys.stderr)
        return 2
    tiers = available if args.tier == "all" else [args.tier]
    tiers = [tier for tier in tiers if tier in available]
    if not tiers:
        print(f"Requested tier is not trained in {artifacts}", file=sys.stderr)
        return 2

    _, _, test_df = load_split()
    if args.sample and args.sample < len(test_df):
        frame = test_df.sample(n=args.sample, random_state=42).reset_index(drop=True)
    else:
        frame = test_df.reset_index(drop=True)

    print("=" * 78)
    print("NEWSCLASS ERROR TAXONOMY AND SLICES")
    print("=" * 78)
    print(f"documents: {len(frame)}  |  artifacts: {artifacts}")

    # Each tier reads a different window, so the truncation category must use the
    # tier's real limit rather than one shared number.
    #
    # bronze is a TF-IDF model: it has no token window and reads the whole
    # document, so "long_document" cannot be a cause of its errors. It gets an
    # effectively infinite limit so the category is never assigned to it.
    UNBOUNDED = 10**9
    default_limits = {"bronze": UNBOUNDED, "silver": 64, "gold": 64}

    results = []
    for tier in tiers:
        limit = args.max_tokens
        if limit is None:
            model = registry.get(tier)
            limit = getattr(model, "max_length", None)
            if limit is None:
                # SilverModel stores its window as `_max_len`, not `max_length`.
                limit = getattr(model, "_max_len", None)
            if limit is None:
                limit = default_limits.get(tier, 64)
        result = evaluate_tier(registry, tier, frame, max_tokens=int(limit), examples=args.examples)
        result["truncates_input"] = int(limit) != UNBOUNDED
        print_tier(result)
        results.append(result)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "documents": len(frame),
        "tiers": results,
    }
    out = Path(args.json_out) if args.json_out else artifacts / "error_taxonomy.json"
    try:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    except OSError as exc:
        print(f"could not write {out}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
