"""Shift-aware confidence calibration.

Gold is well calibrated on AG News and overconfident on modern news: the same
reported 0.90 means ~92% correct in distribution and ~74% correct out of it. A
single temperature cannot serve both regimes, so this module carries two maps
and blends them with a domain gate:

* a character n-gram classifier estimates how unlike the AG News corpus the
  text looks (shift probability ``w``);
* the in-distribution confidence is kept as-is;
* an out-of-distribution Platt map is applied to the same confidence;
* the served confidence is ``(1 - w) * raw + w * mapped``.

Class probabilities are rescaled around the untouched argmax, so calibration can
never change a prediction — only how much the prediction is trusted.

The artifacts are fitted by ``scripts/fit_calibration.py``. If they are missing
(or scikit-learn/joblib are unavailable) the calibrator is a no-op, so a partial
install still serves predictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def rescale(probabilities: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Return probabilities with ``confidence`` on the argmax and the same order.

    The runner-up acts as a floor: a calibrated number below it would reorder the
    classes, so it is clamped rather than allowed to contradict the prediction.
    """
    out = np.array(probabilities, dtype=float, copy=True)
    for index, row in enumerate(out):
        # Mirror np.argmax (first maximum) rather than the last sorted index: on an
        # exact tie the two disagree, and the served label comes from argmax.
        best = int(np.argmax(row))
        runner_up = float(np.max(np.delete(row, best)))
        # A saturated distribution (1.0, 0, 0, 0) would divide by zero below; the
        # epsilon keeps the rescaling finite without moving the argmax.
        top = min(float(row[best]), 1.0 - 1e-9)
        # The runner-up is rescaled to runner_up * (1 - t) / (1 - top), which must
        # stay below t or the label would change. Solving that inequality gives the
        # smallest safe t; the margin keeps the comparison strict.
        floor = runner_up / (1.0 - top + runner_up) + 1e-6
        target = float(np.clip(confidence[index], min(floor, 1.0 - 1e-6), 1.0 - 1e-9))
        # A saturated row has no mass to move, so the only options are a row that no
        # longer sums to one or an unchanged row. Leave it alone.
        if not np.isfinite(target) or 1.0 - float(row[best]) <= 0.0:
            continue
        row *= (1.0 - target) / (1.0 - top)
        row[best] = target
    return out


class ShiftAwareCalibrator:
    """Blends in-distribution and out-of-distribution confidence maps."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.slope: float | None = None
        self.intercept: float | None = None
        self.gate: Any | None = None
        self.metadata: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        path = self.directory / "calibration.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            gate_name = Path(payload["gate"]["path"]).name
            gate_path = self.directory / gate_name
            import joblib

            self.gate = joblib.load(gate_path)
            self.slope = float(payload["platt"]["slope"])
            self.intercept = float(payload["platt"]["intercept"])
            self.metadata = payload
        except Exception:  # pragma: no cover - a broken artifact must not stop serving
            logger.exception("calibration artifact unusable; serving raw confidence")
            self.gate = None
            self.slope = self.intercept = None

    @property
    def enabled(self) -> bool:
        return self.gate is not None and self.slope is not None and self.intercept is not None

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.metadata.get("method"),
            "fitted_at": self.metadata.get("fitted_at"),
            "platt": {"slope": self.slope, "intercept": self.intercept},
        }

    def adjust(self, texts: list[str], probabilities: np.ndarray) -> np.ndarray:
        """Calibrate confidences without changing any predicted label."""
        if not self.enabled or len(texts) == 0:
            return probabilities

        shift = np.asarray(self.gate.predict_proba(texts)[:, 1], dtype=float)
        raw = np.asarray(probabilities, dtype=float).max(axis=1)
        mapped = _sigmoid(self.slope * _logit(raw) + self.intercept)
        confidence = (1.0 - shift) * raw + shift * mapped
        return rescale(probabilities, confidence)
