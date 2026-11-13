"""Prometheus metrics.

We deliberately track a small, high-value set:
  * request count and latency per tier and per predicted class
  * prediction confidence distribution (a proxy for model drift)
  * fallback events (a tier was unavailable)

Label cardinality is bounded: tiers are 3 values, classes are 4.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram, generate_latest

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

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


def metrics_registry() -> bytes:
    """Render metrics in Prometheus text format."""
    return generate_latest()
