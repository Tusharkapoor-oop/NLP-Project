"""Tests for the shift-aware confidence calibrator.

The invariant under every test is the same: calibration may move confidence but
must never move the label. The edge cases below are the ones that can break that
promise — saturated rows, exact ties, extreme confidences, and damaged artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from newsclass.calibration import ShiftAwareCalibrator, rescale


# ---------------------------------------------------------------------------
# rescale
# ---------------------------------------------------------------------------
def test_rescale_preserves_argmax_and_normalises():
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet(np.ones(4), size=500)
    confidence = rng.uniform(0.05, 1.0, size=500)

    out = rescale(probabilities, confidence)

    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)
    assert (out.argmax(axis=1) == probabilities.argmax(axis=1)).all()
    assert (out >= 0.0).all()


def test_rescale_keeps_label_when_confidence_collapses():
    """A calibrated number below the runner-up must not flip the prediction."""
    probabilities = np.array([[0.34, 0.33, 0.20, 0.13]])
    out = rescale(probabilities, np.array([0.10]))
    assert out.argmax() == 0
    assert out.sum() == pytest.approx(1.0)


def test_rescale_leaves_a_saturated_row_untouched():
    """A one-hot row has no mass to move; rescaling it would break normalisation."""
    probabilities = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    out = rescale(probabilities, np.array([0.5, 0.0]))

    assert np.isfinite(out).all()
    assert np.array_equal(out, probabilities)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_rescale_keeps_first_maximum_on_an_exact_tie():
    """Uniform rows tie; the served label is np.argmax, so rescale must agree."""
    probabilities = np.tile(np.array([0.25, 0.25, 0.25, 0.25]), (3, 1))
    assert (probabilities.argmax(axis=1) == 0).all()

    out = rescale(probabilities, np.array([0.4, 0.9, 1.0]))

    assert (out.argmax(axis=1) == probabilities.argmax(axis=1)).all()
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_rescale_clamps_confidence_extremes():
    """Confidence of exactly 0 or 1 must still leave a valid distribution."""
    probabilities = np.tile(np.array([0.4, 0.3, 0.2, 0.1]), (2, 1))
    out = rescale(probabilities, np.array([0.0, 1.0]))

    assert np.isfinite(out).all()
    assert (out.argmax(axis=1) == 0).all()
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)
    # A 0.0 request is lifted to the smallest value that still beats the runner-up.
    assert 0.3 < out[0, 0] < 0.4
    # A 1.0 request is held just below certainty so the row stays a distribution.
    assert out[1, 0] < 1.0


def test_rescale_ignores_a_non_finite_confidence():
    """A NaN from a broken gate must not poison the row."""
    probabilities = np.array([[0.5, 0.3, 0.1, 0.1], [0.2, 0.6, 0.1, 0.1]])
    out = rescale(probabilities, np.array([np.nan, 0.7]))

    assert np.isfinite(out).all()
    assert (out.argmax(axis=1) == probabilities.argmax(axis=1)).all()
    assert np.allclose(out[0], probabilities[0])  # untouched row
    assert out[1, 1] == pytest.approx(0.7, abs=1e-6)


def test_rescale_does_not_mutate_the_input():
    probabilities = np.array([[0.6, 0.2, 0.1, 0.1]])
    before = probabilities.copy()
    rescale(probabilities, np.array([0.3]))
    assert np.array_equal(probabilities, before)


def test_rescale_accepts_an_empty_batch():
    out = rescale(np.empty((0, 4)), np.empty(0))
    assert out.shape == (0, 4)


def test_rescale_argmax_is_stable_under_adversarial_confidence():
    rng = np.random.default_rng(7)
    probabilities = rng.dirichlet(np.ones(4), size=2000)
    confidence = rng.choice([0.0, 1.0, 0.5], size=2000)

    out = rescale(probabilities, confidence)

    assert (out.argmax(axis=1) == probabilities.argmax(axis=1)).all()
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# ShiftAwareCalibrator
# ---------------------------------------------------------------------------
def test_calibrator_is_a_noop_without_artifacts(tmp_path: Path):
    calibrator = ShiftAwareCalibrator(tmp_path)
    probabilities = np.array([[0.7, 0.1, 0.1, 0.1]])
    assert not calibrator.enabled
    assert np.array_equal(calibrator.adjust(["some text"], probabilities), probabilities)


def test_describe_is_honest_when_disabled(tmp_path: Path):
    described = ShiftAwareCalibrator(tmp_path).describe()
    assert described["enabled"] is False
    assert described["platt"] == {"slope": None, "intercept": None}


def test_calibrator_survives_unparseable_json(tmp_path: Path):
    (tmp_path / "calibration.json").write_text("{not json", encoding="utf-8")
    calibrator = ShiftAwareCalibrator(tmp_path)
    assert not calibrator.enabled
    probabilities = np.array([[0.7, 0.1, 0.1, 0.1]])
    assert np.array_equal(calibrator.adjust(["x"], probabilities), probabilities)


def test_calibrator_survives_missing_platt_block(tmp_path: Path):
    (tmp_path / "calibration.json").write_text(json.dumps({"gate": {}}), encoding="utf-8")
    calibrator = ShiftAwareCalibrator(tmp_path)
    assert not calibrator.enabled


def test_calibrator_survives_a_missing_gate_file(tmp_path: Path):
    """The JSON points at a gate that was never copied over."""
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "platt": {"slope": 0.6, "intercept": -0.5},
                "gate": {"path": "absent.joblib"},
            }
        ),
        encoding="utf-8",
    )
    calibrator = ShiftAwareCalibrator(tmp_path)
    assert not calibrator.enabled


def test_gate_path_is_confined_to_the_artifact_directory(tmp_path: Path, monkeypatch):
    """A traversal or absolute path in the JSON must not escape the tier directory."""

    class StubGate:
        def predict_proba(self, texts: list[str]) -> np.ndarray:
            return np.tile([0.1, 0.9], (len(texts), 1))

    loaded: list[Path] = []

    def fake_load(path):
        loaded.append(Path(path))
        return StubGate()

    import joblib

    monkeypatch.setattr(joblib, "load", fake_load)
    (tmp_path / "escape.joblib").write_bytes(b"stub")
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "platt": {"slope": 0.6, "intercept": -0.5},
                "gate": {"path": "../../escape.joblib"},
            }
        ),
        encoding="utf-8",
    )

    calibrator = ShiftAwareCalibrator(tmp_path)

    assert calibrator.enabled
    assert loaded == [tmp_path / "escape.joblib"]


def test_calibrator_reports_enabled_when_artifacts_load(tmp_path: Path, monkeypatch):
    """A present artifact is reported, and adjustment keeps the label stable."""

    class StubGate:
        def predict_proba(self, texts: list[str]) -> np.ndarray:
            return np.tile([0.1, 0.9], (len(texts), 1))

    def fake_load(path):
        return StubGate()

    import joblib

    monkeypatch.setattr(joblib, "load", fake_load)
    (tmp_path / "shift_gate.joblib").write_bytes(b"stub")
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "method": "shift-gated platt",
                "platt": {"slope": 0.6, "intercept": -0.5},
                "gate": {"path": "shift_gate.joblib"},
            }
        ),
        encoding="utf-8",
    )

    calibrator = ShiftAwareCalibrator(tmp_path)
    assert calibrator.enabled

    probabilities = np.array([[0.62, 0.2, 0.1, 0.08]])
    out = calibrator.adjust(["a modern sounding headline"], probabilities)
    assert out.argmax() == probabilities.argmax()
    assert out.sum() == pytest.approx(1.0)
    assert out[0][0] < probabilities[0][0]


def _calibrator_with_shift(tmp_path: Path, monkeypatch, shift: float) -> ShiftAwareCalibrator:
    class StubGate:
        def predict_proba(self, texts: list[str]) -> np.ndarray:
            return np.tile([1.0 - shift, shift], (len(texts), 1))

    import joblib

    monkeypatch.setattr(joblib, "load", lambda path: StubGate())
    (tmp_path / "shift_gate.joblib").write_bytes(b"stub")
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "method": "shift-gated platt",
                "platt": {"slope": 0.5, "intercept": -1.0},
                "gate": {"path": "shift_gate.joblib"},
            }
        ),
        encoding="utf-8",
    )
    return ShiftAwareCalibrator(tmp_path)


def test_zero_shift_leaves_in_distribution_confidence_alone(tmp_path: Path, monkeypatch):
    calibrator = _calibrator_with_shift(tmp_path, monkeypatch, shift=0.0)
    probabilities = np.array([[0.7, 0.15, 0.1, 0.05]])

    out = calibrator.adjust(["an ag news style headline"], probabilities)

    assert np.allclose(out, probabilities)


def test_full_shift_applies_the_platt_map(tmp_path: Path, monkeypatch):
    calibrator = _calibrator_with_shift(tmp_path, monkeypatch, shift=1.0)
    probabilities = np.array([[0.9, 0.05, 0.03, 0.02]])

    out = calibrator.adjust(["a 2026 headline about chip exports"], probabilities)

    assert out.argmax() == 0
    assert out.sum() == pytest.approx(1.0)
    assert out[0, 0] < probabilities[0, 0]  # overconfidence pulled down
    assert out[0, 0] > 0.5  # ... but a 0.9 does not become a coin flip


def test_empty_batch_does_not_touch_the_gate(tmp_path: Path, monkeypatch):
    class ExplodingGate:
        def predict_proba(self, texts):
            raise AssertionError("gate must not be called for an empty batch")

    import joblib

    monkeypatch.setattr(joblib, "load", lambda path: ExplodingGate())
    (tmp_path / "shift_gate.joblib").write_bytes(b"stub")
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "platt": {"slope": 0.5, "intercept": 0.0},
                "gate": {"path": "shift_gate.joblib"},
            }
        ),
        encoding="utf-8",
    )
    calibrator = ShiftAwareCalibrator(tmp_path)
    probabilities = np.empty((0, 4))

    out = calibrator.adjust([], probabilities)

    assert out.shape == (0, 4)
