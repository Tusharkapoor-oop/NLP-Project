"""Fit a shift-aware confidence calibrator for the gold tier.

The problem this solves: gold is calibrated *in distribution* (ECE ~0.03 on AG
News) and badly overconfident *out of distribution* (says 0.90 when it is right
74% of the time on 2026 news). One global temperature cannot fix both, because
the same reported confidence means different things in the two regimes.

So we fit two things:

1. A **shift gate**: a character n-gram classifier that estimates "this text
   looks unlike the AG News corpus". In-distribution text keeps its existing
   (already good) confidence.
2. An **out-of-distribution calibrator**: Platt scaling of gold's own
   confidence, fitted on labelled modern news.

The served confidence is a mixture weighted by the gate, so a document that
looks like AG News is barely touched and a modern-news document gets the OOD
map. Every number printed here is cross-validated out-of-fold: no document's
label is used to calibrate its own confidence.

Usage::

    PYTHONPATH=src python scripts/fit_calibration.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / ".pytest_runs"
OOD_RESULTS = RUNS / "fresh_ood_results.json"
OOD_CACHE = RUNS / "ood_docs.json"
INDIST_CACHE = RUNS / "indist_scores.json"
GATE_CACHE = ROOT / "artifacts" / "gold" / "shift_gate.joblib"
CALIBRATION = ROOT / "artifacts" / "gold" / "calibration.json"
API = os.environ.get("NEWSCLASS_API", "http://127.0.0.1:8012").rstrip("/")

AG_NEWS_URL = "https://huggingface.co/datasets/SetFit/ag_news/resolve/main/test.jsonl"
AG_NEWS_CACHE = RUNS / "agnews_test.jsonl"
AG_SAMPLE = 1500  # in-distribution rows for the gate
INDIST_EVAL = 400  # held-out AG News rows scored through the API
FOLDS = 5


# ---------------------------------------------------------------------------
# small helpers
def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def expected_calibration_error(y: np.ndarray, confidence: np.ndarray, bins: int = 10) -> float:
    """Standard 10-bin ECE: |accuracy - confidence| weighted by bin size."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high)
        if mask.sum() == 0:
            continue
        gap = abs(y[mask].mean() - confidence[mask].mean())
        total += gap * mask.sum() / len(y)
    return float(total)


def brier(y: np.ndarray, confidence: np.ndarray) -> float:
    return float(np.mean((confidence - y) ** 2))


def reliability(y: np.ndarray, confidence: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": f"{low:.1f}-{high:.1f}",
                "n": int(mask.sum()),
                "confidence": round(float(confidence[mask].mean()), 4),
                "accuracy": round(float(y[mask].mean()), 4),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# inputs
def load_ood() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Guardian documents, gold confidences and correctness."""
    rows = json.loads(OOD_RESULTS.read_text(encoding="utf-8"))["tiers"]["gold"]["rows"]
    texts = [r["text"] for r in rows]
    confidence = np.array([r["confidence"] for r in rows], dtype=float)
    correct = np.array([r["label"] == r["gold"] for r in rows], dtype=int)
    return texts, confidence, correct


def load_ag_news() -> tuple[list[str], list[str], np.ndarray]:
    """AG News sample: ``(gate texts, evaluation texts, evaluation label ids)``.

    The JSONL is fetched directly: ``datasets`` spends minutes converting this
    repo to Arrow, and we only need two thousand short strings.
    """
    if AG_NEWS_CACHE.exists():
        payload = AG_NEWS_CACHE.read_text(encoding="utf-8")
    else:
        print(f"downloading {AG_NEWS_URL.rsplit('/', 2)[-2]}/test.jsonl")
        with urllib.request.urlopen(AG_NEWS_URL, timeout=120) as response:
            payload = response.read().decode("utf-8")
        AG_NEWS_CACHE.write_text(payload, encoding="utf-8")

    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    total = min(len(rows), AG_SAMPLE + INDIST_EVAL)
    texts = [rows[i]["text"] for i in range(total)]
    labels = np.array([rows[i]["label"] for i in range(AG_SAMPLE, total)], dtype=int)
    return texts[:AG_SAMPLE], texts[AG_SAMPLE:], labels


def score_api(texts: list[str], tier: str = "gold") -> tuple[np.ndarray, np.ndarray]:
    """Raw confidences and predicted class ids from the running service."""
    labels = json.loads((ROOT / "artifacts" / "registry.json").read_text(encoding="utf-8"))[
        "labels"
    ]
    confidences: list[float] = []
    predicted: list[int] = []
    for start in range(0, len(texts), 64):
        chunk = texts[start : start + 64]
        body = json.dumps({"texts": chunk, "tier": tier}).encode("utf-8")
        request = urllib.request.Request(
            f"{API}/v1/predict/batch",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - environment only
            raise SystemExit(f"cannot reach {API}: {exc}. Start the server first.") from exc
        if payload.get("tier") != tier:
            raise SystemExit(f"refusing to score: server served {payload.get('tier')!r}")
        confidences.extend(p["confidence"] for p in payload["predictions"])
        predicted.extend(labels.index(p["label"]) for p in payload["predictions"])
    return np.array(confidences, dtype=float), np.array(predicted, dtype=int)


# ---------------------------------------------------------------------------
# fitting
def build_gate(indist: list[str], ood: list[str]):
    """Character n-gram domain classifier: AG News (0) vs modern news (1)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    texts = list(indist) + list(ood)
    labels = np.array([0] * len(indist) + [1] * len(ood))
    pipeline = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30_000),
        LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"),
    )
    pipeline.fit(texts, labels)
    return pipeline


def fit_platt(confidence: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    """Logistic regression of correctness on the logit of confidence."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(logit(confidence).reshape(-1, 1), correct)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def apply_platt(confidence: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(slope * logit(confidence) + intercept)))


def cross_validate(confidence: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Out-of-fold calibrated confidences: no document calibrates itself."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    out = np.zeros_like(confidence)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=17)
    for train_index, test_index in splitter.split(confidence.reshape(-1, 1), correct):
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(logit(confidence[train_index]).reshape(-1, 1), correct[train_index])
        out[test_index] = apply_platt(
            confidence[test_index], float(model.coef_[0][0]), float(model.intercept_[0])
        )
    return out


def cross_validate_gate(texts: list[str], labels: np.ndarray) -> np.ndarray:
    """Out-of-fold shift probabilities, so the gate never scores its own training text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline

    out = np.zeros(len(texts))
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=17)
    for train_index, test_index in splitter.split(texts, labels):
        pipeline = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30_000),
            LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"),
        )
        pipeline.fit([texts[i] for i in train_index], labels[train_index])
        out[test_index] = pipeline.predict_proba([texts[i] for i in test_index])[:, 1]
    return out


def summarize(name: str, correct: np.ndarray, confidence: np.ndarray) -> dict:
    y = correct.astype(float)
    high = confidence >= 0.8
    thresholds = {}
    for threshold in (0.6, 0.7, 0.8, 0.9):
        mask = confidence >= threshold
        thresholds[f">={threshold:.1f}"] = {
            "coverage": round(float(mask.mean()), 4),
            "n": int(mask.sum()),
            "accuracy": round(float(correct[mask].mean()), 4) if mask.any() else None,
        }
    result = {
        "name": name,
        "n": int(len(y)),
        "accuracy": round(float(y.mean()), 4),
        "mean_confidence": round(float(confidence.mean()), 4),
        "ece": round(expected_calibration_error(y, confidence), 4),
        "brier": round(brier(y, confidence), 4),
        "nll": round(float(-np.mean(y * np.log(confidence) + (1 - y) * np.log(1 - confidence))), 4),
        "conf_gap": round(
            float(confidence[correct == 1].mean() - confidence[correct == 0].mean()), 4
        ),
        "high_confidence_accuracy": round(float(correct[high].mean()), 4) if high.any() else None,
        "selective": thresholds,
        "reliability": reliability(y, confidence),
    }
    return result


def main() -> int:
    if not OOD_RESULTS.exists():
        raise SystemExit(f"missing {OOD_RESULTS.name}; run scripts/ood_eval.py first")

    print(f"loading out-of-distribution gold rows from {OOD_RESULTS.name}")
    ood_texts, ood_conf, ood_correct = load_ood()
    print(
        f"  n={len(ood_texts)}  raw accuracy={ood_correct.mean():.4f}  mean conf={ood_conf.mean():.4f}"
    )

    print("loading AG News test rows (cached after the first run)")
    gate_texts, indist_eval_texts, indist_truth = load_ag_news()

    # Cached so a later, already-calibrated server cannot double-apply the map.
    if INDIST_CACHE.exists():
        cached = json.loads(INDIST_CACHE.read_text(encoding="utf-8"))
        indist_conf = np.array(cached["confidence"], dtype=float)
        indist_pred = np.array(cached["label"], dtype=int)
        print(f"reusing cached in-distribution scores: {len(indist_conf)}")
    else:
        print(f"scoring {len(indist_eval_texts)} AG News rows through {API} (raw, pre-integration)")
        indist_conf, indist_pred = score_api(indist_eval_texts)
        INDIST_CACHE.write_text(
            json.dumps(
                {"confidence": indist_conf.tolist(), "label": indist_pred.tolist()}, indent=2
            ),
            encoding="utf-8",
        )
    indist_correct = (indist_pred == indist_truth).astype(int)

    # -- gate
    print("fitting shift gate (AG News vs 2026 Guardian)")
    gate = build_gate(gate_texts, ood_texts)
    gate_labels = np.array([0] * len(gate_texts) + [1] * len(ood_texts))
    oof_gate = cross_validate_gate(gate_texts + ood_texts, gate_labels)
    oof_shift = oof_gate[len(gate_texts) :]
    from sklearn.metrics import roc_auc_score

    gate_auc = float(roc_auc_score(gate_labels, oof_gate))
    print(f"  out-of-fold gate AUC={gate_auc:.4f}")

    # -- calibrators
    print("fitting Platt scaling on modern news (5-fold out-of-fold)")
    oof_calibrated_ood = cross_validate(ood_conf, ood_correct)
    slope, intercept = fit_platt(ood_conf, ood_correct)
    print(f"  slope={slope:.3f} intercept={intercept:.3f}")

    indist_shift = gate.predict_proba(indist_eval_texts)[:, 1]
    mixture_ood = (1 - oof_shift) * ood_conf + oof_shift * oof_calibrated_ood
    mixture_indist = (1 - indist_shift) * indist_conf + indist_shift * apply_platt(
        indist_conf, slope, intercept
    )

    report = {
        "ood_raw": summarize("ood raw", ood_correct, ood_conf),
        "ood_calibrated": summarize("ood calibrated (out-of-fold)", ood_correct, mixture_ood),
        "indist_raw": summarize("in-distribution raw", indist_correct, indist_conf),
        "indist_calibrated": summarize(
            "in-distribution with mixture", indist_correct, mixture_indist
        ),
        "gate": {
            "auc": round(gate_auc, 4),
            "mean_shift_ood": round(float(oof_shift.mean()), 4),
            "mean_shift_indist": round(float(indist_shift.mean()), 4),
            "indist_flagged_above_0.5": round(float((indist_shift > 0.5).mean()), 4),
            "ood_flagged_above_0.5": round(float((oof_shift > 0.5).mean()), 4),
        },
        "platt": {"slope": slope, "intercept": intercept},
        "folds": FOLDS,
    }

    for key in ("ood_raw", "ood_calibrated", "indist_raw", "indist_calibrated"):
        row = report[key]
        print(
            f"  {row['name']:<32} acc={row['accuracy']:.4f} conf={row['mean_confidence']:.4f} "
            f"ECE={row['ece']:.4f} Brier={row['brier']:.4f}"
        )
    print(
        f"  gate: AUC={gate_auc:.4f} ood={report['gate']['mean_shift_ood']:.3f} "
        f"indist={report['gate']['mean_shift_indist']:.3f} "
        f"indist flagged={report['gate']['indist_flagged_above_0.5']:.3f}"
    )

    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(gate, GATE_CACHE)
    payload = {
        "schema_version": 1,
        "tier": "gold",
        "method": "shift-gated platt",
        "fitted_at": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "platt": {"slope": slope, "intercept": intercept},
        "gate": {"path": "shift_gate.joblib", "features": "char_wb 3-5 grams"},
        "fit_data": {
            "ood_docs": len(ood_texts),
            "ood_source": "Guardian RSS sections, 2026",
            "ag_news_rows": len(gate_texts),
            "folds": FOLDS,
        },
        "metrics": {
            "ood": {
                "before": {
                    k: report["ood_raw"][k] for k in ("accuracy", "mean_confidence", "ece", "brier")
                },
                "after": {
                    k: report["ood_calibrated"][k]
                    for k in ("accuracy", "mean_confidence", "ece", "brier")
                },
                "selective_after": report["ood_calibrated"]["selective"],
            },
            "indist": {
                "before": {
                    k: report["indist_raw"][k]
                    for k in ("accuracy", "mean_confidence", "ece", "brier")
                },
                "after": {
                    k: report["indist_calibrated"][k]
                    for k in ("accuracy", "mean_confidence", "ece", "brier")
                },
            },
        },
        "notes": (
            "Confidence is usable in both regimes: AG News-like text keeps its in-distribution "
            "calibration, modern news gets a map fitted on labelled 2026 headlines. Fitted on "
            "149 documents, so treat the parameters as provisional and refit as data grows."
        ),
    }
    CALIBRATION.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RUNS / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {CALIBRATION.relative_to(ROOT)} and {GATE_CACHE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
