"""Attach a measured cost-per-1000-requests figure to each tier.

Why this exists
---------------
The tier table has carried a ``cost_per_1k_requests`` column since the first
release, and it has always been ``null``. That is worse than not having the
column: it implies the number exists and nobody filled it in.

Cost is not one thing, so this script reports the two components separately and
then a single comparable figure:

* **compute cost** — derived from the measured effective throughput in
  ``artifacts/benchmark_report.json`` (documents per second at a stated thread
  count). Machine time is priced per core-hour, which is the unit a cloud bill
  actually uses.
* **memory cost** — the resident set the tier needs, priced per GB-hour. This is
  the cost that does not appear in a latency benchmark but does appear on the
  invoice, and it is why a 39 s cold start is not the only problem with gold.

The prices below are **documented assumptions, not facts**. They are set to round
numbers so the output is reproducible and the arithmetic is checkable; anyone can
substitute their own provider's rates. Every report states the assumptions it used
so the figure is never mistaken for a quote.

What this does NOT include, and says so: network egress, storage, idle capacity,
replica count, or engineering time. It is a *serving compute* estimate.

Usage
-----
    python scripts/cost_model.py                        # uses benchmark_report.json
    python scripts/cost_model.py --core-hour 0.05 --gb-hour 0.008
    python scripts/cost_model.py --benchmark bench.json --json-out cost.json

Exit codes: 0 success, 2 if no benchmark exists to derive throughput from.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Documented pricing assumptions, in USD. Round numbers on purpose: the point is
# the ratio between tiers, and a reader must be able to redo the arithmetic.
DEFAULT_CORE_HOUR_USD = 0.04
DEFAULT_GB_HOUR_USD = 0.005

# A serving replica is assumed to stay up for an hour, so memory is charged for
# the full hour at its peak resident footprint.
SECONDS_PER_HOUR = 3600.0


def tier_cost(
    entry: dict,
    *,
    core_hour_usd: float,
    gb_hour_usd: float,
    threads: int,
) -> dict:
    """Cost of serving 1000 documents with one tier, in USD.

    Compute: 1000 docs / (docs per second) = seconds of wall time, spread over the
    worker threads actually used, then priced per core-hour.
    Memory: the tier's peak resident footprint, charged per GB-hour for the wall
    time the request batch occupies the instance.
    """
    throughput = float(entry.get("throughput_docs_per_second") or 0)
    peak_mb = entry.get("peak_rss_mb") or entry.get("rss_delta_mb") or 0.0

    if throughput <= 0:
        return {
            "tier": entry.get("tier", "unknown"),
            "available": False,
            "reason": "no throughput recorded in the benchmark",
        }

    wall_seconds = 1000.0 / throughput
    core_seconds = wall_seconds * threads
    compute_usd = (core_seconds / SECONDS_PER_HOUR) * core_hour_usd
    memory_usd = ((peak_mb / 1024.0) * (wall_seconds / SECONDS_PER_HOUR)) * gb_hour_usd

    return {
        "tier": entry["tier"],
        "available": True,
        "throughput_docs_per_second": round(throughput, 1),
        "wall_seconds_per_1k": round(wall_seconds, 2),
        "core_seconds_per_1k": round(core_seconds, 1),
        "peak_rss_mb": round(float(peak_mb), 1),
        "compute_usd_per_1k": round(compute_usd, 6),
        "memory_usd_per_1k": round(memory_usd, 6),
        "total_usd_per_1k": round(compute_usd + memory_usd, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measured cost per 1,000 requests")
    parser.add_argument(
        "--benchmark",
        default=str(PROJECT_ROOT / "artifacts" / "benchmark_report.json"),
        help="benchmark_report.json to derive throughput from",
    )
    parser.add_argument("--core-hour", type=float, default=DEFAULT_CORE_HOUR_USD)
    parser.add_argument("--gb-hour", type=float, default=DEFAULT_GB_HOUR_USD)
    parser.add_argument("--threads", type=int, default=None, help="defaults to the benchmark's")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the figures into artifacts/registry.json so the API reports them",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    bench_path = Path(args.benchmark)
    if not bench_path.exists():
        print(
            f"no benchmark at {bench_path}\n"
            "run scripts/benchmark.py first: it measures the throughput this needs",
            file=sys.stderr,
        )
        return 2

    benchmark = json.loads(bench_path.read_text(encoding="utf-8"))
    threads = args.threads or int(benchmark.get("threads") or 1)

    print("=" * 78)
    print("NEWSCLASS COST MODEL")
    print("=" * 78)
    print(f"throughput source : {bench_path}")
    print(f"threads           : {threads} (from the benchmark, so the units match)")
    print(
        f"assumptions       : ${args.core_hour:.4f} per core-hour, ${args.gb_hour:.5f} per GB-hour"
    )
    print("                    (documented placeholders — substitute your provider's rates)")
    print("excludes          : egress, storage, idle capacity, replicas, engineer time")
    print()

    results = []
    for entry in benchmark.get("tiers", []):
        cost = tier_cost(
            entry,
            core_hour_usd=args.core_hour,
            gb_hour_usd=args.gb_hour,
            threads=threads,
        )
        results.append(cost)

    available = [r for r in results if r["available"]]
    available.sort(key=lambda r: r["total_usd_per_1k"])
    if not available:
        print("the benchmark recorded no usable throughput", file=sys.stderr)
        return 2

    cheapest = available[0]["total_usd_per_1k"]
    print(
        f"  {'tier':<8} {'docs/s':>9} {'wall s/1k':>10} {'core s/1k':>10} "
        f"{'RSS MB':>8} {'USD/1k':>10} {'x cheapest':>11}"
    )
    for result in available:
        multiple = result["total_usd_per_1k"] / cheapest if cheapest else 0
        print(
            f"  {result['tier']:<8} {result['throughput_docs_per_second']:>9.1f} "
            f"{result['wall_seconds_per_1k']:>10.1f} {result['core_seconds_per_1k']:>10.0f} "
            f"{result['peak_rss_mb']:>8.0f} {result['total_usd_per_1k']:>10.5f} "
            f"{multiple:>10.1f}x"
        )

    print()
    most_expensive = available[-1]
    if len(available) > 1:
        print(
            f"  {most_expensive['tier']} costs "
            f"{most_expensive['total_usd_per_1k'] / cheapest:.0f}x {available[0]['tier']} "
            "per 1,000 requests."
        )
        print(
            "  That multiple is the price of the accuracy gap measured in\n"
            "  artifacts/significance_report.json — the two reports are meant to be\n"
            "  read together, because neither number means much alone."
        )

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "benchmark_source": str(bench_path),
        "threads": threads,
        "assumptions": {
            "core_hour_usd": args.core_hour,
            "gb_hour_usd": args.gb_hour,
            "note": (
                "Documented placeholders, not a quote. Excludes egress, storage, "
                "idle capacity, replica count and engineering time."
            ),
        },
        "tiers": results,
    }

    out = (
        Path(args.json_out) if args.json_out else (PROJECT_ROOT / "artifacts" / "cost_report.json")
    )
    try:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    except OSError as exc:
        print(f"could not write {out}: {exc}", file=sys.stderr)

    if args.apply:
        registry_path = PROJECT_ROOT / "artifacts" / "registry.json"
        if not registry_path.exists():
            print(f"no registry at {registry_path}; skipping --apply", file=sys.stderr)
            return 0
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        by_tier = {r["tier"]: r for r in available}
        updated = 0
        for tier, meta in registry.get("tiers", {}).items():
            if tier in by_tier:
                meta["cost_per_1k_requests"] = by_tier[tier]["total_usd_per_1k"]
                meta["cost_basis"] = report["assumptions"]
                updated += 1
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        print(f"applied cost_per_1k_requests to {updated} tier(s) in {registry_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
