"""In-process statistics: a bounded ring of recent predictions + aggregates.

Prometheus counters are for scraping; the UI needs to *read* numbers back, so
this module keeps a small, thread-safe window of recent events and computes the
aggregates the insights endpoint exposes.

Deliberate design points:

* Bounded memory — the ring never grows past ``max_events``.
* No persistence — this describes *this process's* traffic. The durable record
  of production behaviour is Prometheus; the durable record of human judgement
  is the feedback store.
* Everything is a plain ``dict`` in, plain ``dict`` out, so the API layer adds
  no transformation logic.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from datetime import UTC, datetime
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class PredictionStats:
    """Thread-safe recorder and aggregator for served predictions."""

    def __init__(self, max_events: int = 200) -> None:
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._fallbacks = 0
        self._max_events = max_events

    # -- recording ----------------------------------------------------------
    def record(
        self,
        *,
        tier: str,
        label: str,
        confidence: float,
        latency_ms: float,
        text_chars: int,
        fallback: bool = False,
        batch: bool = False,
    ) -> None:
        event = {
            "ts": time.time(),
            "tier": tier,
            "label": label,
            "confidence": float(confidence),
            "latency_ms": float(latency_ms),
            "text_chars": int(text_chars),
            "fallback": bool(fallback),
            "batch": bool(batch),
        }
        with self._lock:
            self._events.append(event)
            if fallback:
                self._fallbacks += 1

    # -- aggregation --------------------------------------------------------
    def _percentile(self, values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return ordered[int(index)]

    def snapshot(self, recent: int = 25, buckets: int = 12) -> dict[str, Any]:
        """Aggregate the current window into the insights payload."""
        with self._lock:
            events = list(self._events)
            fallbacks = self._fallbacks

        if not events:
            return {
                "window": {"recent_events": 0, "max_events": self._max_events},
                "empty": True,
                "updated_at": _now_iso(),
            }

        labels = Counter(e["label"] for e in events)
        tiers = Counter(e["tier"] for e in events)
        confidences = [e["confidence"] for e in events]
        latencies = [e["latency_ms"] for e in events]
        low_conf = [c for c in confidences if c < 0.60]

        # Per-label time buckets for the sparkline: split the window evenly.
        oldest = events[0]["ts"]
        newest = events[-1]["ts"]
        span = max(newest - oldest, 1e-9)
        bucket_counts: dict[str, list[int]] = {label: [0] * buckets for label in labels}
        for e in events:
            slot = min(buckets - 1, int((e["ts"] - oldest) / span * buckets))
            bucket_counts[e["label"]][slot] += 1

        total = len(events)
        recent_conf = confidences[-recent:] if len(confidences) >= recent else confidences

        return {
            "window": {"recent_events": total, "max_events": self._max_events},
            "empty": False,
            "updated_at": _now_iso(),
            "label_mix": [
                {"label": label, "count": count, "share": round(count / total, 3)}
                for label, count in labels.most_common()
            ],
            "tier_usage": [{"tier": tier, "count": count} for tier, count in tiers.most_common()],
            "confidence": {
                "mean": round(sum(confidences) / total, 4),
                "mean_recent": round(sum(recent_conf) / len(recent_conf), 4),
                "low_confidence_share": round(len(low_conf) / total, 3),
            },
            "latency": {
                "p50_ms": round(self._percentile(latencies, 0.50), 1),
                "p95_ms": round(self._percentile(latencies, 0.95), 1),
                "mean_ms": round(sum(latencies) / total, 1),
            },
            "fallbacks": fallbacks,
            "series_labels": {label: series for label, series in bucket_counts.items()},
        }
