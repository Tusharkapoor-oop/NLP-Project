"""Prometheus metrics with a zero-dependency fallback.

We deliberately track a small, high-value set:
  * request count and latency per tier and per predicted class
  * prediction confidence distribution (a proxy for model drift)
  * fallback events (a tier was unavailable)
  * unhandled errors by endpoint

Label cardinality is bounded: tiers are 3 values, classes are 4.

Why the fallback
----------------
``prometheus-client`` is a serving dependency, but the service must still boot
on a machine where it is missing (a reviewer's laptop, a minimal container, a
constrained lab image). Rather than failing at import time — which takes down
the whole API — this module provides a small thread-safe implementation of the
same surface (``Counter``/``Histogram``/``generate_latest``) and renders valid
Prometheus exposition text. When the real client is installed it is used and
nothing below the ``try`` block changes.
"""

from __future__ import annotations

import threading
from typing import Any

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

try:  # pragma: no cover - exercised by whichever branch the environment hits
    from prometheus_client import Counter, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PROMETHEUS_AVAILABLE = False

    class _Child:
        """One label combination of a metric."""

        __slots__ = ("_parent", "_labels")

        def __init__(self, parent: _Metric, labels: dict[str, str]) -> None:
            self._parent = parent
            self._labels = labels

        def inc(self, amount: float = 1.0) -> None:
            self._parent._inc(self._labels, amount)

        def observe(self, value: float) -> None:
            self._parent._observe(self._labels, value)

    class _Metric:
        """Minimal stand-in for a prometheus_client Counter/Histogram."""

        def __init__(
            self,
            name: str,
            documentation: str,
            labelnames: tuple[str, ...] = (),
            buckets: tuple[float, ...] | None = None,
        ) -> None:
            self.name = name
            self.documentation = documentation
            self.labelnames = labelnames
            self.buckets = buckets
            self._lock = threading.Lock()
            self._counts: dict[tuple[str, ...], float] = {}
            self._sums: dict[tuple[str, ...], float] = {}
            self._bucket_counts: dict[tuple[str, ...], list[float]] = {}

        # -- public API ----------------------------------------------------
        def labels(self, **labels: str) -> _Child:
            ordered = tuple(str(labels.get(name, "")) for name in self.labelnames)
            return _Child(self, dict(zip(self.labelnames, ordered, strict=False)))

        # -- internals -----------------------------------------------------
        def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
            return tuple(labels.get(name, "") for name in self.labelnames)

        def _inc(self, labels: dict[str, str], amount: float) -> None:
            key = self._key(labels)
            with self._lock:
                self._counts[key] = self._counts.get(key, 0.0) + amount

        def _observe(self, labels: dict[str, str], value: float) -> None:
            key = self._key(labels)
            with self._lock:
                self._counts[key] = self._counts.get(key, 0.0) + 1
                self._sums[key] = self._sums.get(key, 0.0) + float(value)
                if self.buckets:
                    slots = self._bucket_counts.setdefault(key, [0.0] * len(self.buckets))
                    for index, bound in enumerate(self.buckets):
                        if value <= bound:
                            slots[index] += 1

        def _render_labels(self, key: tuple[str, ...]) -> str:
            if not self.labelnames:
                return ""
            parts = [f'{name}="{value}"' for name, value in zip(self.labelnames, key, strict=False)]
            return "{" + ",".join(parts) + "}"

        def render(self) -> list[str]:
            lines = [
                f"# HELP {self.name} {self.documentation}",
                f"# TYPE {self.name} {'histogram' if self.buckets else 'counter'}",
            ]
            with self._lock:
                keys = sorted(self._counts)
                for key in keys:
                    label_text = self._render_labels(key)
                    if self.buckets:
                        cumulative = 0.0
                        for index, bound in enumerate(self.buckets):
                            cumulative = self._bucket_counts.get(key, [0.0] * len(self.buckets))[
                                index
                            ]
                            lines.append(
                                f"{self.name}_bucket{self._with_le(label_text, bound)} {cumulative}"
                            )
                        lines.append(
                            f"{self.name}_bucket{self._with_le(label_text, '+Inf')} "
                            f"{self._counts.get(key, 0.0)}"
                        )
                        lines.append(f"{self.name}_sum{label_text} {self._sums.get(key, 0.0)}")
                        lines.append(f"{self.name}_count{label_text} {self._counts.get(key, 0.0)}")
                    else:
                        lines.append(f"{self.name}{label_text} {self._counts[key]}")
            return lines

        @staticmethod
        def _with_le(label_text: str, bound: float | str) -> str:
            if not label_text:
                return f'{{le="{bound}"}}'
            return label_text[:-1] + f',le="{bound}"}}'

    _METRICS: list[_Metric] = []

    def Counter(  # noqa: N802 - mirrors prometheus_client's API
        name: str, documentation: str, labelnames: tuple[str, ...] = (), **_kwargs: Any
    ) -> _Metric:
        metric = _Metric(name, documentation, labelnames)
        _METRICS.append(metric)
        return metric

    def Histogram(  # noqa: N802 - mirrors prometheus_client's API
        name: str,
        documentation: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] = (),
        **_kwargs: Any,
    ) -> _Metric:
        metric = _Metric(name, documentation, labelnames, buckets=tuple(buckets))
        _METRICS.append(metric)
        return metric

    def generate_latest() -> bytes:
        lines: list[str] = []
        for metric in _METRICS:
            lines.extend(metric.render())
        return ("\n".join(lines) + "\n").encode("utf-8")


PREDICTIONS = Counter(
    "newsclass_predictions_total",
    "Number of prediction requests served.",
    ["tier", "label"],
)

LATENCY = Histogram(
    "newsclass_prediction_latency_seconds",
    "Server-side inference latency.",
    ["tier"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

CONFIDENCE = Histogram(
    "newsclass_prediction_confidence",
    "Distribution of the winning class probability (drift signal).",
    ["tier"],
    buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

FALLBACKS = Counter(
    "newsclass_tier_fallbacks_total",
    "Requests served by a fallback tier because the requested one was unavailable.",
    ["requested_tier", "served_tier"],
)

ERRORS = Counter(
    "newsclass_errors_total",
    "Unhandled errors by endpoint.",
    ["endpoint", "error_type"],
)

FEEDBACK_COUNTER = Counter(
    "newsclass_feedback_total",
    "Human corrections recorded through the feedback endpoint.",
    ["corrected_label"],
)


def observe_prediction(*, tier: str, labels: list[str], index: int, duration_s: float) -> None:
    """Record one served prediction."""
    label = labels[index] if 0 <= index < len(labels) else "unknown"
    PREDICTIONS.labels(tier=tier, label=label).inc()
    LATENCY.labels(tier=tier).observe(duration_s)


def observe_confidence(*, tier: str, confidence: float) -> None:
    CONFIDENCE.labels(tier=tier).observe(confidence)


def observe_fallback(*, requested_tier: str, served_tier: str) -> None:
    FALLBACKS.labels(requested_tier=requested_tier, served_tier=served_tier).inc()


def observe_error(*, endpoint: str, error_type: str) -> None:
    ERRORS.labels(endpoint=endpoint, error_type=error_type).inc()


def observe_feedback(*, corrected_label: str) -> None:
    FEEDBACK_COUNTER.labels(corrected_label=corrected_label).inc()


def metrics_registry() -> bytes:
    """Render metrics in Prometheus text format."""
    payload = generate_latest()
    return payload if isinstance(payload, bytes) else str(payload).encode("utf-8")


def metrics_backend() -> str:
    """Which implementation is live. Surfaced by ``/v1/model`` for audits."""
    return "prometheus-client" if PROMETHEUS_AVAILABLE else "builtin-fallback"
