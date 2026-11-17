"""Statistical significance for tier comparisons: error bars and paired tests.

Why this exists
---------------
The tier table quotes three point estimates — bronze 0.901, silver 0.898,
gold 0.918. Point estimates alone cannot answer the question a reviewer will
actually ask: **is that ordering real, or is it run-to-run noise?**

The README previously said variance was "about ±0.5 accuracy points" and asked
the reader not to over-read small gaps. That is honest but incomplete: it leaves
the central claim of the project — that gold is worth its cost — unquantified.
This script replaces the caveat with measurement.

What it computes
----------------
1. **Bootstrap confidence intervals** (default 2000 resamples) for accuracy and
   macro-F1 per tier. The interval is the honest presentation of a single number.
2. **McNemar's test** on every tier pair. Because all tiers see the *same*
   documents, the correct comparison is paired, not independent: McNemar asks
   whether the disagreements between two models are lopsided. A chi-square
   approximation is used when the discordant count is large, and an exact
   binomial test when it is small (the usual rule: < 25 discordant pairs).
3. **Chance-corrected agreement**: Cohen's kappa and Matthews correlation
   coefficient (MCC). Accuracy alone flatters a model on imbalanced data; MCC is
   the single-number summary that does not.

Reading the output
------------------
`delta` is the accuracy difference (A − B) and `p_value` is the probability of
seeing a lopsided disagreement split that extreme if the two models were equally
good. A small p means the gap is unlikely to be chance. `significant` applies
the chosen alpha (default 0.05) with a Bonferroni correction across the number
of pairs tested, because testing every pair inflates the false-positive rate.

Usage
-----
    python scripts/significance.py                      # all tiers, full test set
    python scripts/significance.py --sample 1000
    python scripts/significance.py --alpha 0.01 --resamples 5000
    python scripts/significance.py --require-order      # gate: best must beat worst

Exit codes: 0 pass, 1 gate failed, 2 nothing to evaluate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from itertools import combinations
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
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)

from newsclass.config import env_artifacts_dir  # noqa: E402
from newsclass.registry import ModelRegistry  # noqa: E402
from newsclass.train import LABEL_NAMES, load_split  # noqa: E402

# Cheapest to most capable. Used for the ordering gate, so it is explicit rather
# than inferred from accuracy (which would make the gate self-fulfilling).
TIER_COST_ORDER = ("bronze", "silver", "gold")


def bootstrap_interval(
    gold: np.ndarray,
    predicted: np.ndarray,
    *,
    resamples: int,
    seed: int,
    alpha: float,
) -> dict:
    """Percentile bootstrap CI for accuracy and macro-F1.

    Resampling *documents* (with replacement) is the right unit here: the test
    set is a sample of documents, and we want the uncertainty in the metric
    caused by which documents happened to be drawn. macro-F1 is recomputed on
    each resample because it is not a simple average of per-document scores.
    """
    generator = np.random.default_rng(seed)
    n = len(gold)
    correct = (predicted == gold).astype(np.float64)

    accuracies = np.empty(resamples, dtype=np.float64)
    f1s = np.empty(resamples, dtype=np.float64)

    for index in range(resamples):
        take = generator.integers(0, n, size=n)
        accuracies[index] = correct[take].mean()
        f1s[index] = f1_score(gold[take], predicted[take], average="macro", zero_division=0)

    lower = alpha / 2 * 100
    upper = (1 - alpha / 2) * 100
    return {
        "resamples": resamples,
        "accuracy_ci": [
            round(float(np.percentile(accuracies, lower)), 4),
            round(float(np.percentile(accuracies, upper)), 4),
        ],
        "accuracy_std": round(float(accuracies.std(ddof=1)), 4),
        "macro_f1_ci": [
            round(float(np.percentile(f1s, lower)), 4),
            round(float(np.percentile(f1s, upper)), 4),
        ],
        "macro_f1_std": round(float(f1s.std(ddof=1)), 4),
    }


def mcnemar(
    predicted_a: np.ndarray,
    predicted_b: np.ndarray,
    gold: np.ndarray,
) -> dict:
    """Paired comparison of two models over identical documents.

    Only the *discordant* pairs carry information: documents where both models
    are right, or both wrong, cannot distinguish them. b and c count those
    discordant cases, and the test asks whether b is far enough from c to reject
    "the two models are equally good".
    """
    correct_a = predicted_a == gold
    correct_b = predicted_b == gold

    both_correct = int(np.sum(correct_a & correct_b))
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))
    discordant = only_a + only_b

    if discordant == 0:
        return {
            "only_a_correct": only_a,
            "only_b_correct": only_b,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "discordant": 0,
            "test": "none",
            "statistic": 0.0,
            "p_value": 1.0,
            "note": "the two models never disagree: no evidence either way",
        }

    # Exact binomial is correct for small discordant counts; the chi-square
    # approximation with continuity correction is the standard large-sample form.
    if discordant < 25:
        from scipy.stats import binomtest

        result = binomtest(only_a, discordant, 0.5, alternative="two-sided")
        return {
            "only_a_correct": only_a,
            "only_b_correct": only_b,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "discordant": discordant,
            "test": "exact binomial",
            "statistic": None,
            "p_value": round(float(result.pvalue), 6),
            "note": "small discordant count: exact test used",
        }

    from scipy.stats import chi2

    statistic = (abs(only_a - only_b) - 1) ** 2 / discordant
    p_value = float(chi2.sf(statistic, df=1))
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "test": "chi-square (continuity corrected)",
        "statistic": round(float(statistic), 4),
        "p_value": round(p_value, 6),
        "note": "",
    }


def evaluate_tier(registry: ModelRegistry, tier: str, texts: list[str], gold: list[int]) -> dict:
    """Predict once and keep the predictions: every later test reuses them."""
    model = registry.get(tier)
    started = time.perf_counter()
    probabilities = model.predict_proba(texts)
    elapsed_ms = (time.perf_counter() - started) * 1000

    predicted = probabilities.argmax(axis=1)
    gold_array = np.asarray(gold, dtype=np.int64)

    return {
        "tier": tier,
        "documents": len(texts),
        "accuracy": round(float(accuracy_score(gold_array, predicted)), 4),
        "macro_f1": round(float(f1_score(gold_array, predicted, average="macro")), 4),
        "cohen_kappa": round(float(cohen_kappa_score(gold_array, predicted)), 4),
        "matthews_corrcoef": round(float(matthews_corrcoef(gold_array, predicted)), 4),
        "mean_confidence": round(float(probabilities.max(axis=1).mean()), 4),
        "eval_ms_total": round(elapsed_ms, 1),
        "_predicted": predicted,
        "_gold": gold_array,
    }


def build_sample(test_df, sample: int):
    """Stratified sample, matching accuracy_report.py so numbers stay comparable."""
    if sample and sample < len(test_df):
        import pandas as pd

        per_class = max(1, sample // len(LABEL_NAMES))
        parts = [
            group.sample(n=min(per_class, len(group)), random_state=42)
            for _, group in test_df.groupby("label")
        ]
        return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return test_df.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Significance testing and error bars")
    parser.add_argument("--tier", default="all", choices=["all", "bronze", "silver", "gold"])
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="0 = the whole test split (maximum power, recommended here)",
    )
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-order",
        action="store_true",
        help=(
            "gate: fail unless each more expensive tier is significantly better "
            "than the cheapest tier it is compared against"
        ),
    )
    parser.add_argument("--artifacts-dir", default=str(env_artifacts_dir()))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    registry = ModelRegistry(artifacts)
    available = [t for t in registry.available_tiers() if t in TIER_COST_ORDER]
    if not available:
        print(f"No trained tiers found in {artifacts}", file=sys.stderr)
        return 2
    if args.tier != "all":
        available = [t for t in available if t == args.tier]
        if not available:
            print(f"tier {args.tier} is not trained in {artifacts}", file=sys.stderr)
            return 2

    print("=" * 78)
    print("NEWSCLASS SIGNIFICANCE REPORT")
    print("=" * 78)

    _, _, test_df = load_split()
    frame = build_sample(test_df, args.sample)
    texts = frame["text"].tolist()
    gold = frame["label"].tolist()
    print(f"documents   : {len(texts)} (stratified when sampled, seed 42)")
    print(f"resamples   : {args.resamples} for each confidence interval")
    print(f"alpha       : {args.alpha}")

    results: dict[str, dict] = {}
    for tier in available:
        started = time.time()
        tier_result = evaluate_tier(registry, tier, texts, gold)
        interval = bootstrap_interval(
            tier_result["_gold"],
            tier_result["_predicted"],
            resamples=args.resamples,
            seed=args.seed,
            alpha=args.alpha,
        )
        tier_result.update(interval)
        results[tier] = tier_result
        print(f"  evaluated {tier:<7} in {time.time() - started:6.1f}s")

    # -- per-tier summary -------------------------------------------------
    print("\n" + "-" * 78)
    print("PER-TIER ESTIMATES WITH BOOTSTRAP CONFIDENCE INTERVALS")
    print("-" * 78)
    label = f"{1 - args.alpha:.0%}"
    ci_header = f"{label} CI"
    print(
        f"  {'tier':<8} {'accuracy':>9}  {ci_header:>18}  {'macro-F1':>9}  {'kappa':>7}  {'MCC':>7}"
    )
    for tier in available:
        r = results[tier]
        ci = r["accuracy_ci"]
        print(
            f"  {tier:<8} {r['accuracy']:>9.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]  "
            f"{r['macro_f1']:>9.4f}  {r['cohen_kappa']:>7.4f}  {r['matthews_corrcoef']:>7.4f}"
        )

    # -- pairwise McNemar -------------------------------------------------
    pairs = list(combinations(available, 2))
    correction = max(1, len(pairs))
    effective_alpha = args.alpha / correction

    print("\n" + "-" * 78)
    print("PAIRED COMPARISONS (McNemar) — same documents, so the test is paired")
    print("-" * 78)
    print(
        f"  Bonferroni: {len(pairs)} pair(s) tested, so a gap counts as significant "
        f"only below alpha={effective_alpha:.4f}"
    )
    print()
    print(f"  {'A':<8} {'B':<8} {'delta':>8} {'discordant':>11} {'test':>28} {'p':>9}  verdict")

    comparisons = []
    for tier_a, tier_b in pairs:
        a, b = results[tier_a], results[tier_b]
        stats = mcnemar(a["_predicted"], b["_predicted"], a["_gold"])
        delta = round(a["accuracy"] - b["accuracy"], 4)
        significant = stats["p_value"] < effective_alpha
        if not significant:
            verdict = "no reliable difference"
        else:
            better = tier_a if delta > 0 else tier_b
            verdict = f"{better} is better"
        comparisons.append(
            {
                "tier_a": tier_a,
                "tier_b": tier_b,
                "delta_accuracy": delta,
                "significant": significant,
                "p_value": stats["p_value"],
                "effective_alpha": round(effective_alpha, 6),
                **{k: v for k, v in stats.items() if k != "note"},
                "note": stats.get("note", ""),
            }
        )
        print(
            f"  {tier_a:<8} {tier_b:<8} {delta:>+8.4f} {stats['discordant']:>11} "
            f"{stats['test']:>28} {stats['p_value']:>9.4f}  {verdict}"
        )

    # -- ordering gate ----------------------------------------------------
    failures: list[str] = []
    if args.require_order:
        print("\n" + "-" * 78)
        print("ORDERING GATE")
        print("-" * 78)
        ordered = [t for t in TIER_COST_ORDER if t in available]
        for cheaper, dearer in zip(ordered, ordered[1:], strict=False):
            match = next(
                (c for c in comparisons if {c["tier_a"], c["tier_b"]} == {cheaper, dearer}),
                None,
            )
            if match is None:
                continue
            cheaper_acc = results[cheaper]["accuracy"]
            dearer_acc = results[dearer]["accuracy"]
            beats = match["significant"] and dearer_acc > cheaper_acc
            mark = "ok  " if beats else "FAIL"
            print(
                f"  {mark} {dearer} vs {cheaper}: delta={dearer_acc - cheaper_acc:+.4f} "
                f"p={match['p_value']:.4f}  "
                f"{'significantly better' if beats else 'NOT significantly better'}"
            )
            if not beats:
                failures.append(
                    f"{dearer} must be significantly better than {cheaper} to justify "
                    f"its higher cost, but p={match['p_value']:.4f} >= {effective_alpha:.4f}"
                )

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "documents": len(texts),
        "resamples": args.resamples,
        "alpha": args.alpha,
        "bonferroni_correction": correction,
        "effective_alpha": round(effective_alpha, 6),
        "seed": args.seed,
        "tiers": [
            {k: v for k, v in results[t].items() if not k.startswith("_")} for t in available
        ],
        "comparisons": comparisons,
        "require_order": args.require_order,
        "passed": not failures,
        "failures": failures,
    }

    out = Path(args.json_out) if args.json_out else artifacts / "significance_report.json"
    try:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    except OSError as exc:
        print(f"could not write {out}: {exc}", file=sys.stderr)

    if failures:
        print("\nORDERING GATE FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nThis is not a bug in the models. It means the project must not claim "
            "that ordering: either report the tiers as statistically tied, or make "
            "the claim only where the test supports it.",
            file=sys.stderr,
        )
        return 1

    print("\nSignificance report complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
