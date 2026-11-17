"""Operational benchmark: latency percentiles, throughput, memory and cold start.

Why this exists
---------------
`scripts/evaluate.py` measures per-tier latency as one number — total wall time
over 100 documents, divided. That is enough to compare tiers roughly, but it is
not enough to run the service:

* **A mean hides the tail.** A user notices the p99, not the mean. A model with a
  fine average and a 4-second p99 is a bad model for interactive use.
* **Single-request latency is not throughput.** Serving one request at a time
  measures a desktop script, not a service. Real traffic is concurrent, and the
  interesting question is how latency degrades under load.
* **Memory is a deployment constraint.** Three tiers warm at once is a different
  container size than one. This is the number that decides the instance type.
* **Cold start decides autoscaling.** If loading a tier takes 39 s, a new replica
  cannot serve that tier for 39 s after boot — which is why preloading exists.

What it measures
----------------
For each tier, over a held-out sample:
  * latency percentiles: p50, p90, p95, p99, min, max, mean
  * **effective throughput** under N concurrent worker threads
  * **peak resident memory** (RSS delta) attributed to that tier, measured by
    sampling the process while it runs
  * **cold-start time** — a fresh subprocess that loads only that tier, so the
    number is not polluted by other tiers already resident

Concurrency uses a thread pool. Every ML runtime here releases the GIL while
computing, so threads measure real parallel throughput; the report notes the
thread count so the number is never read as "one core" or "infinite clients".

Usage
-----
    python scripts/benchmark.py                        # all tiers, 8 threads
    python scripts/benchmark.py --tier bronze --threads 16
    python scripts/benchmark.py --documents 500 --skip-cold-start
    python scripts/benchmark.py --json-out bench.json

Exit codes: 0 pass, 1 a configured budget was exceeded, 2 nothing to benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from newsclass.config import env_artifacts_dir  # noqa: E402
from newsclass.registry import ModelRegistry  # noqa: E402
from newsclass.train import load_split  # noqa: E402


def _rss_mb() -> float | None:
    """Resident set size of this process, in MB, or None if unavailable.

    psutil is used when present because it is accurate on every platform; the
    Windows fallback reads the same counter through the OS API so the benchmark
    still reports memory on a machine without psutil.
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    if sys.platform == "win32":  # pragma: no cover - platform specific
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            return counters.WorkingSetSize / 1e6
        except Exception:  # noqa: BLE001 - never let measurement break the run
            return None
    return None


class MemorySampler:
    """Samples RSS on a background thread to find the peak, not just the end."""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mb = 0.0
        self.available = _rss_mb() is not None

    def _run(self) -> None:
        while not self._stop.is_set():
            current = _rss_mb()
            if current is not None and current > self.peak_mb:
                self.peak_mb = current
            self._stop.wait(self.interval)

    def __enter__(self) -> MemorySampler:
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def cold_start_seconds(tier: str, artifacts: Path) -> float | None:
    """Load one tier in a fresh process and time it.

    A subprocess is required: once a tier is resident, re-loading it is a cache
    hit and would report a meaningless near-zero. This is the number that decides
    whether a replica can answer immediately after boot.
    """
    code = (
        "import os, sys, time\n"
        f"sys.path.insert(0, r'{PROJECT_ROOT / 'src'}')\n"
        "from pathlib import Path\n"
        "from newsclass.registry import ModelRegistry\n"
        f"started = time.perf_counter()\n"
        f"registry = ModelRegistry(Path(r'{artifacts}'))\n"
        f"registry.get('{tier}')\n"
        "print(time.perf_counter() - started)\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None
    if completed.returncode != 0:
        return None
    try:
        return round(float(completed.stdout.strip().splitlines()[-1]), 2)
    except (ValueError, IndexError):
        return None


def benchmark_tier(
    registry: ModelRegistry,
    tier: str,
    texts: list[str],
    *,
    threads: int,
    repeats: int,
) -> dict:
    """Measure one tier: percentiles, concurrent throughput and peak memory."""
    baseline_mb = _rss_mb() or 0.0

    # --- single-request latency, one at a time ---------------------------
    model = registry.get(tier)  # ensure loaded before measuring
    latencies_ms: list[float] = []
    for text in texts[: min(len(texts), 120)]:
        started = time.perf_counter()
        model.predict_proba([text])
        latencies_ms.append((time.perf_counter() - started) * 1000)

    # --- throughput under concurrency ------------------------------------
    chunks: list[list[str]] = [[] for _ in range(threads)]
    for index, text in enumerate(texts):
        chunks[index % threads].append(text)

    def worker(chunk: list[str]) -> int:
        if not chunk:
            return 0
        model.predict_proba(chunk)
        return len(chunk)

    # Warm all threads once so the measured window is steady-state.
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(worker, chunks))

    with MemorySampler() as sampler:
        started = time.perf_counter()
        for _ in range(max(1, repeats)):
            with ThreadPoolExecutor(max_workers=threads) as pool:
                served = sum(pool.map(worker, chunks))
        elapsed = time.perf_counter() - started

    throughput = served / elapsed if elapsed > 0 else 0.0
    peak_mb = sampler.peak_mb

    ordered = sorted(latencies_ms)

    def pct(value: float) -> float:
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, max(0, round(value * (len(ordered) - 1))))
        return round(ordered[int(index)], 2)

    return {
        "tier": tier,
        "documents_per_single_request": len(latencies_ms),
        "latency_ms": {
            "min": round(ordered[0], 2) if ordered else 0.0,
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": round(ordered[-1], 2) if ordered else 0.0,
            "mean": round(statistics.fmean(ordered), 2) if ordered else 0.0,
        },
        "threads": threads,
        "repeats": repeats,
        "throughput_docs_per_second": round(throughput, 1),
        "throughput_docs_per_minute": round(throughput * 60, 0),
        "peak_rss_mb": round(peak_mb, 1) if peak_mb else None,
        "rss_delta_mb": round(max(0.0, peak_mb - baseline_mb), 1) if peak_mb else None,
        "memory_available": sampler.available,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Operational benchmark for every tier")
    parser.add_argument("--tier", default="all", choices=["all", "bronze", "silver", "gold"])
    parser.add_argument("--documents", type=int, default=300, help="documents per tier")
    parser.add_argument("--threads", type=int, default=8, help="concurrent workers")
    parser.add_argument("--repeats", type=int, default=2, help="passes over the corpus")
    parser.add_argument("--skip-cold-start", action="store_true")
    parser.add_argument(
        "--max-p99-ms",
        type=float,
        default=None,
        help="optional budget: fail if any tier's p99 exceeds this",
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

    print("=" * 78)
    print("NEWSCLASS OPERATIONAL BENCHMARK")
    print("=" * 78)

    _, _, test_df = load_split()
    sample = test_df.sample(n=min(args.documents, len(test_df)), random_state=42)
    texts = sample["text"].tolist()
    print(f"documents per tier : {len(texts)}")
    print(f"concurrency        : {args.threads} worker threads x {args.repeats} passes")
    print(f"memory sampling    : {'on' if _rss_mb() is not None else 'unavailable'}")
    if args.threads > (os.cpu_count() or 1):
        print(
            f"note               : {args.threads} threads exceeds "
            f"{os.cpu_count()} CPUs, so cores are oversubscribed"
        )

    results = []
    for tier in tiers:
        print(f"\nbenchmarking {tier} ...", flush=True)
        result = benchmark_tier(registry, tier, texts, threads=args.threads, repeats=args.repeats)
        if not args.skip_cold_start:
            cold = cold_start_seconds(tier, artifacts)
            result["cold_start_seconds"] = cold
        results.append(result)
        lat = result["latency_ms"]
        print(f"  p50={lat['p50']:>8.2f} ms  p95={lat['p95']:>8.2f} ms  p99={lat['p99']:>8.2f} ms")
        print(
            f"  throughput={result['throughput_docs_per_second']:>8.1f} docs/s "
            f"({result['throughput_docs_per_minute']:.0f}/min) at {args.threads} threads"
        )
        if result.get("peak_rss_mb"):
            print(
                f"  peak RSS={result['peak_rss_mb']:.0f} MB "
                f"(+{result['rss_delta_mb']:.0f} MB while this tier ran)"
            )
        if result.get("cold_start_seconds") is not None:
            print(f"  cold start={result['cold_start_seconds']:.1f} s")

    failures: list[str] = []
    if args.max_p99_ms is not None:
        for result in results:
            if result["latency_ms"]["p99"] > args.max_p99_ms:
                failures.append(
                    f"{result['tier']}: p99 {result['latency_ms']['p99']} ms > {args.max_p99_ms} ms"
                )

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "documents": len(texts),
        "threads": args.threads,
        "repeats": args.repeats,
        "cpu_count": os.cpu_count(),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "max_p99_ms": args.max_p99_ms,
        "tiers": results,
        "passed": not failures,
        "failures": failures,
    }

    out = Path(args.json_out) if args.json_out else artifacts / "benchmark_report.json"
    try:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    except OSError as exc:
        print(f"could not write {out}: {exc}", file=sys.stderr)

    if failures:
        print("\nLATENCY BUDGET FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
