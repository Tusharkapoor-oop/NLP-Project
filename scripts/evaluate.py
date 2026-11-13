"""End-to-end evaluation of the trained tiers against the real test set.

This is the script you run before a release. It answers one question:
"does the artifact we are about to ship actually perform?"

Usage:
    python scripts/evaluate.py                 # evaluate every ready tier
    python scripts/evaluate.py --tier bronze   # one tier
    python scripts/evaluate.py --gate 0.85     # fail if macro-F1 < 0.85
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running as a plain script without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Keep Hugging Face caches inside the project so the script works on machines
# where the user profile directory is read-only.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sklearn.metrics import accuracy_score, classification_report, f1_score

from newsclass.config import env_artifacts_dir
from newsclass.registry import ModelRegistry
from newsclass.train import LABEL_NAMES, load_split


def evaluate_tier(registry: ModelRegistry, tier: str, sample: int, artifacts: Path) -> dict:
    _, _, test_df = load_split()
    if sample and sample < len(test_df):
        test_df = test_df.sample(n=sample, random_state=42).reset_index(drop=True)

    model = registry.get(tier)
    texts = test_df["text"].tolist()
    gold = test_df["label"].tolist()

    started = time.perf_counter()
    probabilities = model.predict_proba(texts)
    elapsed_ms = (time.perf_counter() - started) * 1000

    predicted = probabilities.argmax(axis=1).tolist()
    accuracy = float(accuracy_score(gold, predicted))
    macro_f1 = float(f1_score(gold, predicted, average="macro"))

    per_100 = elapsed_ms / max(1, len(texts)) * 100
    confidence = float(probabilities.max(axis=1).mean())

    report = classification_report(
        gold,
        predicted,
        target_names=LABEL_NAMES,
        digits=3,
        output_dict=True,
        zero_division=0,
    )

    return {
        "tier": tier,
        "n_evaluated": len(texts),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "mean_confidence": round(confidence, 4),
        "total_latency_ms": round(elapsed_ms, 1),
        "ms_per_100_docs": round(per_100, 1),
        "per_class_f1": {label: round(report[label]["f1-score"], 3) for label in LABEL_NAMES},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained tiers on the held-out test set")
    parser.add_argument("--tier", default="all", choices=["all", "bronze", "silver", "gold"])
    parser.add_argument("--sample", type=int, default=1000, help="0 = use the whole test set")
    parser.add_argument("--gate", type=float, default=None, help="minimum acceptable macro-F1")
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
    results = []
    failures = []

    for tier in tiers:
        if tier not in available:
            print(f"SKIP {tier}: no artifacts")
            continue
        try:
            result = evaluate_tier(registry, tier, args.sample, artifacts)
        except Exception as exc:
            print(f"FAIL {tier}: {exc}", file=sys.stderr)
            failures.append(tier)
            continue
        results.append(result)
        print(
            f"{result['tier']:<7} acc={result['accuracy']:.4f} "
            f"macro_f1={result['macro_f1']:.4f} "
            f"conf={result['mean_confidence']:.3f} "
            f"{result['ms_per_100_docs']:.0f} ms/100 docs"
        )

    if args.json_out and results:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    if args.gate is not None:
        for result in results:
            if result["macro_f1"] < args.gate:
                print(
                    f"GATE FAILED: {result['tier']} macro_f1={result['macro_f1']} "
                    f"< {args.gate}",
                    file=sys.stderr,
                )
                return 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
