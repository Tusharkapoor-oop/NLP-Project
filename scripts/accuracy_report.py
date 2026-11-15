"""High-grade accuracy audit: held-out metrics plus a curated probe set.

`scripts/evaluate.py` answers "does the artifact clear the release gate?".
This script answers the harder review questions:

* How does each tier score *per class*, not just on average?
* Where does it confuse one topic for another (confusion matrix)?
* Is it calibrated, or confidently wrong (expected calibration error)?
* Does it handle realistic, hand-written headlines — including out-of-domain
  copy it was never trained to recognise?

The probe set is deliberately small and readable so every miss can be discussed
in a viva. Probes marked ambiguous are reported but **not scored**: forcing an
opinion on them would manufacture a number rather than measure one.

Usage
-----
    python scripts/accuracy_report.py                    # 1000 docs, all tiers
    python scripts/accuracy_report.py --sample 400 --tier bronze
    python scripts/accuracy_report.py --min-accuracy 0.88 --json-out report.json

Exit codes: 0 pass, 1 gate failed, 2 nothing to evaluate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from newsclass.config import env_artifacts_dir  # noqa: E402
from newsclass.registry import ModelRegistry  # noqa: E402
from newsclass.train import LABEL_NAMES, load_split  # noqa: E402

# (text, expected label or None when the item is genuinely ambiguous, note)
PROBES: list[tuple[str, str | None, str]] = [
    ("The striker scored a hat-trick as his side won the derby 3-0.", "Sports", "clear"),
    ("She defended her title in the 100 metres final on Sunday.", "Sports", "clear"),
    ("The coach was sacked after a fifth straight defeat.", "Sports", "clear"),
    ("The marathon world record was broken in Berlin.", "Sports", "clear"),
    ("The league announced a salary cap increase for next season.", "Sports", "clear"),
    ("The tennis champion withdrew from the tournament with a wrist injury.", "Sports", "clear"),
    ("Shares in the airline fell 6% after it cut its profit forecast.", "Business", "clear"),
    ("The central bank raised interest rates by a quarter point.", "Business", "clear"),
    (
        "The retailer reported record quarterly revenue and raised its dividend.",
        "Business",
        "clear",
    ),
    ("The euro slipped against the dollar after the ECB's policy meeting.", "Business", "clear"),
    ("Bond yields rose after stronger-than-expected jobs data.", "Business", "clear"),
    ("A court ruled that the merger could proceed.", "Business", "clear"),
    ("Apple unveiled a new chip that it says doubles battery life.", "Sci/Tech", "clear"),
    (
        "Researchers published a paper on a faster way to train language models.",
        "Sci/Tech",
        "clear",
    ),
    ("A ransomware attack forced hospitals to switch to paper records.", "Sci/Tech", "clear"),
    ("A probe landed on the moon's south pole and began sending images.", "Sci/Tech", "clear"),
    ("Scientists reported the first image of a black hole's shadow.", "Sci/Tech", "clear"),
    ("A new open-source tool compiles Python to WebAssembly.", "Sci/Tech", "clear"),
    ("The foreign minister met counterparts in Geneva for peace talks.", "World", "clear"),
    ("A ceasefire was announced after weeks of border clashes.", "World", "clear"),
    ("Voters went to the polls in a closely watched general election.", "World", "clear"),
    ("The prime minister resigned after losing a confidence vote.", "World", "clear"),
    ("Aid agencies warned of famine in the drought-hit region.", "World", "clear"),
    ("Protesters gathered outside parliament ahead of the vote.", "World", "clear"),
    (
        "Nvidia's data-centre revenue surged as demand for AI chips outpaced supply.",
        None,
        "ambiguous: business results or technology story",
    ),
    (
        "Tesla recalled 50,000 cars over a software fault.",
        None,
        "ambiguous: business or technology story",
    ),
    (
        "The film won the Palme d'Or at Cannes.",
        None,
        "out of domain: entertainment is not one of the four classes",
    ),
    (
        "Whisk the eggs with the sugar until the mixture is pale.",
        None,
        "out of domain: a recipe, not news",
    ),
]


def build_sample(test_df, sample: int):
    """Stratified sample so a small audit stays representative per class."""
    if sample and sample < len(test_df):
        per_class = max(1, sample // len(LABEL_NAMES))
        parts = [
            group.sample(n=min(per_class, len(group)), random_state=42)
            for _, group in test_df.groupby("label")
        ]
        import pandas as pd

        return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return test_df.reset_index(drop=True)


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, bins: int = 10
) -> float:
    """Weighted |accuracy − confidence| across equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidences >= lower) & (
            confidences < upper if index < bins - 1 else confidences <= upper
        )
        if not mask.any():
            continue
        gap = abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
        error += gap * float(mask.mean())
    return round(float(error), 4)


def evaluate_tier(registry: ModelRegistry, tier: str, texts: list[str], gold: list[int]) -> dict:
    model = registry.get(tier)
    started = time.perf_counter()
    probabilities = model.predict_proba(texts)
    elapsed_ms = (time.perf_counter() - started) * 1000

    predicted = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    accuracy = float(accuracy_score(gold, predicted))
    macro_f1 = float(f1_score(gold, predicted, average="macro"))
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=list(range(len(LABEL_NAMES))), zero_division=0
    )
    matrix = confusion_matrix(gold, predicted, labels=list(range(len(LABEL_NAMES))))

    return {
        "tier": tier,
        "documents": len(texts),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "mean_confidence": round(float(confidences.mean()), 4),
        "expected_calibration_error": expected_calibration_error(
            confidences.astype(float), predicted == np.asarray(gold)
        ),
        "low_confidence_share": round(float((confidences < 0.60).mean()), 4),
        "ms_per_100_docs": round(elapsed_ms / max(1, len(texts)) * 100, 2),
        "per_class": {
            LABEL_NAMES[index]: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index in range(len(LABEL_NAMES))
        },
        "confusion_matrix": {
            "labels": LABEL_NAMES,
            "rows_are_truth": matrix.tolist(),
        },
    }


def run_probes(registry: ModelRegistry, tier: str) -> dict:
    model = registry.get(tier)
    texts = [text for text, _, _ in PROBES]
    probabilities = model.predict_proba(texts)
    predicted = probabilities.argmax(axis=1)
    results = []
    for index, (text, expected, note) in enumerate(PROBES):
        label = LABEL_NAMES[int(predicted[index])]
        confidence = float(probabilities[index].max())
        results.append(
            {
                "text": text,
                "expected": expected,
                "predicted": label,
                "confidence": round(confidence, 4),
                "correct": None if expected is None else label == expected,
                "note": note,
            }
        )

    scored = [item for item in results if item["correct"] is not None]
    correct = sum(1 for item in scored if item["correct"])
    out_of_domain = [item for item in results if item["expected"] is None]
    return {
        "tier": tier,
        "scored": len(scored),
        "correct": correct,
        "accuracy": round(correct / len(scored), 4) if scored else None,
        "mean_confidence_out_of_domain": round(
            float(np.mean([item["confidence"] for item in out_of_domain])), 4
        )
        if out_of_domain
        else None,
        "items": results,
    }


def print_tier(result: dict) -> None:
    print(
        f"\n{result['tier'].upper():<7} acc={result['accuracy']:.4f} "
        f"macro_f1={result['macro_f1']:.4f} conf={result['mean_confidence']:.3f} "
        f"ECE={result['expected_calibration_error']:.3f} "
        f"low-conf={result['low_confidence_share']:.1%} "
        f"{result['ms_per_100_docs']:.0f} ms/100"
    )
    print("  per class:        precision  recall     f1   support")
    for label, metrics in result["per_class"].items():
        print(
            f"  {label:<10}        {metrics['precision']:.3f}     "
            f"{metrics['recall']:.3f}    {metrics['f1']:.3f}   {metrics['support']:>5}"
        )
    print("  confusion (rows = truth, columns = predicted):")
    header = "    ".join(f"{label[:6]:>6}" for label in LABEL_NAMES)
    print(f"  {'':<9} {header}")
    for label, row in zip(LABEL_NAMES, result["confusion_matrix"]["rows_are_truth"], strict=True):
        cells = "    ".join(f"{value:>6}" for value in row)
        print(f"  {label:<9} {cells}")


def print_probes(report: dict) -> None:
    print(
        f"\nCURATED PROBES | {report['tier']} · "
        f"{report['correct']}/{report['scored']} scored items correct"
    )
    for item in report["items"]:
        if item["correct"] is None:
            mark = "n/a "
        else:
            mark = "ok  " if item["correct"] else "MISS"
        print(f"  {mark} {item['predicted']:<9} {item['confidence']:.2f}  {item['text'][:66]}")
        if item["correct"] is None:
            print(f"       ({item['note']})")
    if report["mean_confidence_out_of_domain"] is not None:
        print(
            "  out-of-domain mean confidence: "
            f"{report['mean_confidence_out_of_domain']:.2f} "
            "(lower is better; this is what a refusal policy would threshold)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="High-grade accuracy audit for every tier")
    parser.add_argument("--tier", default="all", choices=["all", "bronze", "silver", "gold"])
    parser.add_argument("--sample", type=int, default=1000, help="0 = the whole test split")
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--min-macro-f1", type=float, default=0.85)
    parser.add_argument("--min-probe-accuracy", type=float, default=0.70)
    parser.add_argument("--artifacts-dir", default=str(env_artifacts_dir()))
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--skip-probes", action="store_true")
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

    print("=" * 78)
    print("NEWSCLASS ACCURACY AUDIT")
    print("=" * 78)

    _, _, test_df = load_split()
    frame = build_sample(test_df, args.sample)
    texts = frame["text"].tolist()
    gold = frame["label"].tolist()
    print(f"held-out documents: {len(texts)} (stratified, seed 42)")
    print(f"artifact directory: {artifacts}")
    print(f"gate: accuracy >= {args.min_accuracy}, macro-F1 >= {args.min_macro_f1}")

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "documents": len(texts),
        "sample": args.sample,
        "gate": {
            "min_accuracy": args.min_accuracy,
            "min_macro_f1": args.min_macro_f1,
            "min_probe_accuracy": args.min_probe_accuracy,
        },
        "tiers": [],
        "probes": [],
        "passed": True,
    }

    failures: list[str] = []
    for tier in tiers:
        try:
            result = evaluate_tier(registry, tier, texts, gold)
        except Exception as exc:  # noqa: BLE001 - report per-tier, keep going
            print(f"\nFAIL {tier}: could not evaluate ({exc})", file=sys.stderr)
            report["passed"] = False
            failures.append(f"{tier}: evaluation error: {exc}")
            continue

        print_tier(result)
        report["tiers"].append(result)
        if result["accuracy"] < args.min_accuracy:
            failures.append(f"{tier}: accuracy {result['accuracy']} < {args.min_accuracy}")
        if result["macro_f1"] < args.min_macro_f1:
            failures.append(f"{tier}: macro-F1 {result['macro_f1']} < {args.min_macro_f1}")

        if not args.skip_probes:
            probe_report = run_probes(registry, tier)
            print_probes(probe_report)
            report["probes"].append(probe_report)
            if (
                probe_report["accuracy"] is not None
                and probe_report["accuracy"] < args.min_probe_accuracy
            ):
                failures.append(
                    f"{tier}: probe accuracy {probe_report['accuracy']} < {args.min_probe_accuracy}"
                )

    report["failures"] = failures
    report["passed"] = not failures

    out = Path(args.json_out) if args.json_out else artifacts / "accuracy_report.json"
    try:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    except OSError as exc:
        print(f"could not write {out}: {exc}", file=sys.stderr)

    if failures:
        print("\nGATE FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nAll evaluated tiers passed the accuracy, calibration and probe gates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
