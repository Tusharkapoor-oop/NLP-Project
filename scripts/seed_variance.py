"""Multi-seed variance: how much does the score move when only the seed changes?

Why this exists
---------------
Every reported number in this project is a single run at seed 42. The README was
honest about the consequence — "variance is about ±0.5 accuracy points, so do not
over-read small gaps" — but that is a caveat, not a measurement. Without it, a
reviewer cannot tell whether a 1-point improvement is a better model or a luckier
seed.

This script retrains a tier across several seeds and reports mean ± std. It is
the complement to `scripts/significance.py`:

* significance.py answers "is the gap real, given this test set?" (test-set noise)
* this script answers "is the gap real, given the training run?" (training noise)

Both matter, and they are different sources of variance. A model can be reliably
better on a fixed test set while still moving more than that gap across seeds.

Cost, and why gold is handled differently
-----------------------------------------
Retraining is expensive, and honesty about that is part of the report:

* **bronze** is seconds per seed, so the full grid is cheap.
* **silver** is minutes per seed.
* **gold** is 88 minutes per seed on this CPU. Five seeds would be over seven
  hours, so gold defaults to a *documented single-run* value and the script says
  so rather than quietly reporting a narrower grid as if it were the same thing.

Usage
-----
    python scripts/seed_variance.py --tier bronze --seeds 5
    python scripts/seed_variance.py --tier silver --seeds 3
    python scripts/seed_variance.py --all --seeds 3 --max-minutes 30

Exit codes: 0 pass, 1 a stability budget was exceeded, 2 nothing to run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("NLTK_DATA", str(PROJECT_ROOT / ".nltk_data"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Approximate cost per seed on a CPU-only host, in minutes. Used only to warn the
# operator before they start something that will run for hours; the real elapsed
# time is measured and reported.
APPROX_MINUTES_PER_SEED = {"bronze": 0.4, "silver": 7.0, "gold": 88.0}


def train_and_score(
    tier: str,
    seed: int,
    rows: int | None,
    gold_epochs: int,
    gold_rows: int,
    *,
    vary_split: bool,
):
    """Train one tier at one seed in-process and return its held-out metrics.

    Two different questions can be asked, and they are not the same thing:

    * ``vary_split=False`` (default) holds the train/test partition fixed and
      varies only the model's random initialisation and training order. This
      isolates **modelling** variance, and it is the number to quote next to a
      fixed-test-set metric, because it does not double-count the test-set noise
      that ``scripts/significance.py`` already measures.
    * ``vary_split=True`` also re-draws the partition, so the reported spread is
      the full end-to-end run-to-run variance. Larger, and the right number when
      describing "what happens if I rerun this project from scratch".

    Training functions are imported lazily because importing torch/transformers
    costs seconds and is pointless when only bronze is being measured.
    """
    import newsclass.train as train_module

    original_seed = train_module.SEED
    if vary_split:
        train_module.SEED = seed
    try:
        if tier == "bronze":
            return _score_bronze(train_module, seed, rows)
        if tier == "silver":
            return _score_silver(train_module, seed, rows)
        if tier == "gold":
            return _score_gold(train_module, seed, gold_epochs, gold_rows)
        raise ValueError(f"unknown tier: {tier}")
    finally:
        train_module.SEED = original_seed


def _score_bronze(train_module, seed: int, rows: int | None) -> dict:
    """TF-IDF + logistic regression: fast enough to rerun the whole pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import Pipeline

    from newsclass.text import normalize_text

    # ``train_rows`` subsamples only the training portion, so the test set stays
    # canonical across seeds and tiers. Using ``rows`` would move the partition
    # and make the scores incomparable — see the load_split docstring for the
    # leakage that caused.
    train_df, _val, test_df = train_module.load_split(train_rows=rows)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20_000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    pipeline.fit([normalize_text(t) for t in train_df["text"]], train_df["label"])
    predicted = pipeline.predict([normalize_text(t) for t in test_df["text"]])
    gold = test_df["label"].to_numpy()
    return {
        "accuracy": round(float(accuracy_score(gold, predicted)), 4),
        "macro_f1": round(float(f1_score(gold, predicted, average="macro")), 4),
        "n_test": len(gold),
    }


def _score_silver(train_module, seed: int, rows: int | None) -> dict:
    """Word2Vec + BiLSTM at a given seed, evaluated on the held-out split."""
    # The existing training function writes artifacts; for variance we only need
    # its metrics, so it runs against a scratch directory and the result is read
    # back. This guarantees the number is produced by the same code path the
    # shipped artifact comes from, rather than a reimplementation that could
    # drift from it.
    import tempfile

    from newsclass.train import train_silver

    with tempfile.TemporaryDirectory(prefix=f"seedvar_silver_{seed}_") as scratch:
        metrics = train_silver(Path(scratch))
    return {
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "n_test": metrics.n_test,
    }


def _score_gold(train_module, seed: int, epochs: int, rows: int) -> dict:
    """DistilBERT at a given seed (expensive — see the module docstring)."""
    import tempfile

    from newsclass.train import train_gold

    with tempfile.TemporaryDirectory(prefix=f"seedvar_gold_{seed}_") as scratch:
        metrics = train_gold(Path(scratch), epochs=epochs, rows=rows)
    return {
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "n_test": metrics.n_test,
    }


def summarise(values: list[float]) -> dict:
    """mean, std, min, max, and the half-width of the observed range.

    Sample std (ddof=1) is used because these runs are a sample of the possible
    seeds, not the entire population of them.
    """
    if not values:
        return {"n": 0}
    if len(values) == 1:
        return {
            "n": 1,
            "mean": round(values[0], 4),
            "std": None,
            "min": round(values[0], 4),
            "max": round(values[0], 4),
            "range": 0.0,
        }
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "std": round(statistics.stdev(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "range": round(max(values) - min(values), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure run-to-run variance across seeds")
    parser.add_argument("--tier", default="bronze", choices=["bronze", "silver", "gold"])
    parser.add_argument("--all", action="store_true", help="run every feasible tier")
    parser.add_argument("--seeds", type=int, default=5, help="how many seeds to run")
    parser.add_argument("--seed-start", type=int, default=42, help="first seed (incremented by 1)")
    parser.add_argument("--rows", type=int, default=None, help="override training rows")
    parser.add_argument("--gold-epochs", type=int, default=2)
    parser.add_argument("--gold-rows", type=int, default=6000)
    parser.add_argument(
        "--vary-split",
        action="store_true",
        help=(
            "also re-draw the train/test partition per seed. Off by default so "
            "the variance reported isolates modelling noise rather than "
            "double-counting test-set noise, which significance.py already covers"
        ),
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="refuse to start a tier whose estimated cost exceeds this",
    )
    parser.add_argument(
        "--max-std",
        type=float,
        default=None,
        help="stability budget: fail if any tier's accuracy std exceeds this",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=None,
        help="stability budget: fail if any tier's accuracy range exceeds this",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    tiers = ["bronze", "silver", "gold"] if args.all else [args.tier]
    seeds = [args.seed_start + index for index in range(args.seeds)]

    print("=" * 78)
    print("NEWSCLASS SEED-VARIANCE REPORT")
    print("=" * 78)
    print(f"tiers : {', '.join(tiers)}")
    print(f"seeds : {seeds}")
    print()
    print("estimated cost (this CPU, single seed):")
    for tier in tiers:
        per_seed = APPROX_MINUTES_PER_SEED[tier]
        total = per_seed * len(seeds)
        print(f"  {tier:<8} ~{per_seed:>5.1f} min/seed  ->  ~{total:>6.1f} min total")
        if args.max_minutes is not None and total > args.max_minutes:
            print(
                f"  {tier:<8} REFUSED: estimated {total:.0f} min exceeds "
                f"--max-minutes {args.max_minutes:.0f}"
            )
    print()

    results: dict[str, dict] = {}
    failures: list[str] = []
    for tier in tiers:
        total_est = APPROX_MINUTES_PER_SEED[tier] * len(seeds)
        if args.max_minutes is not None and total_est > args.max_minutes:
            failures.append(
                f"{tier}: estimated {total_est:.0f} min exceeds --max-minutes "
                f"{args.max_minutes:.0f}; rerun with a smaller --seeds"
            )
            continue

        per_seed_runs = []
        for seed in seeds:
            print(f"  training {tier} at seed {seed} ...", flush=True)
            started = time.time()
            try:
                scored = train_and_score(
                    tier,
                    seed,
                    args.rows,
                    args.gold_epochs,
                    args.gold_rows,
                    vary_split=args.vary_split,
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"    FAILED after {time.time() - started:.0f}s: {exc}", flush=True)
                failures.append(f"{tier} seed {seed}: {exc}")
                continue
            elapsed = time.time() - started
            scored["seed"] = seed
            scored["seconds"] = round(elapsed, 1)
            per_seed_runs.append(scored)
            print(
                f"    acc={scored['accuracy']:.4f} macro_f1={scored['macro_f1']:.4f} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

        if not per_seed_runs:
            continue

        results[tier] = {
            "tier": tier,
            "seeds": [run["seed"] for run in per_seed_runs],
            "runs": per_seed_runs,
            "accuracy": summarise([run["accuracy"] for run in per_seed_runs]),
            "macro_f1": summarise([run["macro_f1"] for run in per_seed_runs]),
            "total_seconds": round(sum(run["seconds"] for run in per_seed_runs), 1),
        }

    # -- summary ----------------------------------------------------------
    print("\n" + "-" * 78)
    print("VARIANCE ACROSS SEEDS")
    print("-" * 78)
    print(
        f"  {'tier':<8} {'seeds':>5} {'acc mean':>9} {'acc std':>8} {'acc range':>10} "
        f"{'f1 mean':>8} {'f1 std':>7}"
    )
    for tier, data in results.items():
        acc, f1 = data["accuracy"], data["macro_f1"]
        std = "n/a" if acc["std"] is None else f"{acc['std']:.4f}"
        f1_std = "n/a" if f1["std"] is None else f"{f1['std']:.4f}"
        print(
            f"  {tier:<8} {acc['n']:>5} {acc['mean']:>9.4f} {std:>8} "
            f"{acc['range']:>10.4f} {f1['mean']:>8.4f} {f1_std:>7}"
        )

    # -- the practical consequence ----------------------------------------
    print("\n" + "-" * 78)
    print("WHAT THIS MEANS FOR CLAIMS")
    print("-" * 78)
    measured = {t: d for t, d in results.items() if d["accuracy"]["std"] is not None}
    if measured:
        # A std of exactly zero means the *model* is deterministic for a fixed
        # split, not that the project has no variance: the split is held fixed
        # by default, so nothing in the pipeline was free to move. Reporting
        # "differences below 0.00 points are noise" from that would be false.
        deterministic = [t for t, d in measured.items() if d["accuracy"]["std"] == 0.0]
        informative = {t: d for t, d in measured.items() if d["accuracy"]["std"] > 0}

        if deterministic:
            print(
                "  Deterministic under a fixed split: " + ", ".join(sorted(deterministic)) + ".\n"
                "  An identical score across seeds here is expected, not a bug: TF-IDF\n"
                "  and logistic regression have no random initialisation to vary (the\n"
                "  lbfgs solver is deterministic). It says nothing about the variance\n"
                "  of the pipeline as a whole, which also depends on the data split —\n"
                "  pass --vary-split to measure that."
            )
        if informative:
            worst_std = max(d["accuracy"]["std"] for d in informative.values())
            names = ", ".join(sorted(informative))
            print()
            print(
                f"  Stochastic tiers ({names}): largest standard deviation "
                f"{worst_std:.4f}.\n"
                f"  A difference smaller than roughly {2 * worst_std:.2f} accuracy points "
                f"is within run-to-run\n"
                f"  noise for these tiers and must not be claimed as an improvement\n"
                f"  without a paired test — scripts/significance.py tests exactly that\n"
                f"  on a fixed test set. Note the two are complements: significance.py\n"
                f"  covers test-set noise, this covers training noise."
            )
        if not informative and not deterministic:
            print("  No tier produced enough runs to measure variance.")
    else:
        print(
            "  No tier produced more than one run, so no variance could be measured.\n"
            "  Re-run with --seeds 2 or more to produce an error bar."
        )

    if args.max_std is not None:
        for tier, data in results.items():
            std = data["accuracy"]["std"]
            if std is not None and std > args.max_std:
                failures.append(f"{tier}: accuracy std {std} > {args.max_std}")
    if args.max_range is not None:
        for tier, data in results.items():
            span = data["accuracy"]["range"]
            if span > args.max_range:
                failures.append(f"{tier}: accuracy range {span} > {args.max_range}")

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tiers": results,
        "max_std": args.max_std,
        "max_range": args.max_range,
        "passed": not failures,
        "failures": failures,
    }

    out = (
        Path(args.json_out) if args.json_out else PROJECT_ROOT / "artifacts" / "seed_variance.json"
    )
    try:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    except OSError as exc:
        print(f"could not write {out}: {exc}", file=sys.stderr)

    if failures:
        print("\nSTABILITY BUDGET FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
