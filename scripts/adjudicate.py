"""Rescore a tier against reviewer-adjudicated labels.

The OOD evaluation uses the publisher's section as ground truth, which is wrong
for some documents: a video-game feature filed under Business, or a newsletter
promo filed under Sport. This script applies a recorded adjudication ledger to
those disputed items and reports accuracy with the corrected labels, excluding
documents that have no defensible single label.

Only disputed documents appear in the ledger. Everything else keeps its section
label, which is stated in the output rather than silently assumed.

Usage::

    python scripts/adjudicate.py --tier gold
    python scripts/adjudicate.py --results .pytest_runs/fresh_ood_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / ".pytest_runs" / "fresh_ood_results.json"
DEFAULT_LEDGER = ROOT / "artifacts" / "adjudication_ood.jsonl"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_ledger(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"missing adjudication ledger: {path}")
    ledger = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            ledger[entry["sha1"]] = entry
        except (json.JSONDecodeError, KeyError) as exc:
            # A hand-edited ledger will get a stray comma eventually; name the line
            # instead of failing with a bare decoder message.
            raise SystemExit(f"{path.name}:{number} is not a valid ledger entry ({exc})") from exc
    return ledger


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def prf(rows: list[dict], label: str) -> tuple[float, float, float, int]:
    tp = sum(1 for r in rows if r["truth"] == label and r["label"] == label)
    fp = sum(1 for r in rows if r["truth"] != label and r["label"] == label)
    fn = sum(1 for r in rows if r["truth"] == label and r["label"] != label)
    support = sum(1 for r in rows if r["truth"] == label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, support


def score(rows: list[dict], ledger: dict[str, dict]) -> dict:
    kept, excluded = [], []
    for row in rows:
        entry = ledger.get(sha1(row["text"]))
        if entry is None:
            kept.append({**row, "truth": row["gold"], "source": "section"})
        elif entry["adjudicated"] == "EXCLUDE":
            excluded.append({**row, "reason": entry["note"]})
        else:
            kept.append({**row, "truth": entry["adjudicated"], "source": "adjudicated"})

    n = len(kept)
    correct = sum(1 for r in kept if r["label"] == r["truth"])
    low, high = wilson(correct, n)

    # Per-tier accounting of where the errors came from.
    adjudicated = [r for r in kept if r["source"] == "adjudicated"]
    relabelled = [r for r in adjudicated if r["truth"] != r["gold"]]
    # Published errors that vanish: the section label was wrong and the model was right.
    noise = [r for r in relabelled if r["label"] == r["truth"]]
    # Errors the section label had hidden: the model matched the wrong section.
    revealed = [r for r in relabelled if r["label"] == r["gold"]]
    genuine = [r for r in kept if r["label"] != r["truth"]]

    f1s = [prf(kept, label)[2] for label in LABELS]
    return {
        "n": n,
        "excluded": len(excluded),
        "covered_by_ledger": len(adjudicated) + len(excluded),
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "ci95": [low, high],
        "macro_f1": sum(f1s) / len(f1s),
        "per_class": {label: prf(kept, label) for label in LABELS},
        "relabelled": len(relabelled),
        "label_noise_removed": len(noise),
        "errors_revealed": len(revealed),
        "published_errors": sum(1 for r in rows if r["label"] != r["gold"]),
        "errors_excluded": sum(1 for r in excluded if r["label"] != r["gold"]),
        "final_errors": len(genuine),
        "excluded_items": excluded,
        "rows": kept,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rescore tiers with adjudicated labels")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--tier", default="all", choices=["all", "bronze", "silver", "gold"])
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(f"missing results file: {args.results}; run scripts/ood_eval.py first")
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    ledger = load_ledger(args.ledger)

    tiers = ["bronze", "silver", "gold"] if args.tier == "all" else [args.tier]
    out: dict = {"ledger": str(args.ledger.name), "tiers": {}}

    print(f"adjudication ledger: {len(ledger)} disputed documents")
    for tier in tiers:
        rows = payload["tiers"][tier]["rows"]
        raw_correct = sum(1 for r in rows if r["label"] == r["gold"])
        result = score(rows, ledger)
        out["tiers"][tier] = result
        low, high = result["ci95"]
        print(f"\n=== {tier} ===")
        print(
            f"  as published : {raw_correct}/{len(rows)} = {raw_correct / len(rows):.4f}"
            f"   (section labels, all documents)"
        )
        print(
            f"  adjudicated  : {result['correct']}/{result['n']} = {result['accuracy']:.4f}"
            f"   95% CI [{low:.3f}, {high:.3f}]  macro-F1={result['macro_f1']:.4f}"
        )
        print(
            f"    error accounting: {result['published_errors']} published errors "
            f"- {result['label_noise_removed']} pure label noise "
            f"- {result['errors_excluded']} excluded items "
            f"+ {result['errors_revealed']} revealed "
            f"= {result['final_errors']} genuine errors in {result['n']}"
        )
        print("  per class:")
        for label in LABELS:
            precision, recall, f1, support = result["per_class"][label]
            print(f"    {label:<9} P={precision:.3f} R={recall:.3f} F1={f1:.3f} n={support}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {k: v for k, v in out.items() if k != "tiers"}
                | {
                    "tiers": {
                        tier: {k: v for k, v in res.items() if k != "rows"}
                        for tier, res in out["tiers"].items()
                    }
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
