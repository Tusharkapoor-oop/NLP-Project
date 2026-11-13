"""Shared pytest fixtures.

The API is exercised through ``TestClient`` against a *temporary* artifacts
directory so tests never depend on (or damage) the real trained models.
Where a trained tier is required, a small synthetic model is written on the fly.

Why not pytest's ``tmp_path``?
-----------------------------
``tmp_path`` works fine on ordinary machines, but its numbered-directory
factory performs recursive garbage-collection sweeps at fixture setup. On
restricted environments (CI sandboxes, locked-down lab images) those sweeps
are refused and *every* fixture-based test fails with a misleading
"Access is denied" before the test code even runs. A plain directory created
by the test process itself — and removed afterwards, errors ignored — has
none of those failure modes, so that is what these fixtures use.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# Disposable directories live inside the project; the sandbox guarantees this
# tree is writable, unlike a system temp directory that may be locked down.
RUNS_DIR = PROJECT_ROOT / ".pytest_runs"
_CASE_COUNTER = itertools.count()


def _fresh_case_dir() -> Path:
    """Create an isolated scratch directory for one test case."""
    RUNS_DIR.mkdir(exist_ok=True)
    case = RUNS_DIR / f"case-{os.getpid()}-{next(_CASE_COUNTER)}-{uuid.uuid4().hex[:8]}"
    case.mkdir()
    return case


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


def _remove_quietly(directory: Path) -> None:
    """Best-effort cleanup; a locked file must never fail a test run."""
    shutil.rmtree(directory, ignore_errors=True)


def _write_tinydf_bronze(directory: Path) -> None:
    """Create a genuinely trained (tiny) TF-IDF model so the API has something to serve."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    texts = [
        "stocks rise as the market gains on strong earnings",
        "markets fall after the central bank raises rates",
        "the striker scored twice in the cup final",
        "the team won the match in extra time",
        "apple launches a new laptop with a faster chip",
        "the software update fixes a critical bug",
        "the president met foreign ministers for talks",
        "the government announced a new policy on trade",
    ] * 4
    labels = [2, 2, 1, 1, 3, 3, 0, 0] * 4

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(texts, labels)

    import joblib

    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, directory / "model.joblib")
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "tier": "bronze",
                "model_name": "tfidf+logreg-test",
                "model_version": "test",
                "accuracy": 0.9,
                "macro_f1": 0.9,
                "labels": LABELS,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def artifacts_dir():
    """A fresh artifacts directory containing only a bronze tier."""
    case = _fresh_case_dir()
    target = case / "artifacts"
    _write_tinydf_bronze(target / "bronze")
    (target / "registry.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "labels": LABELS,
                "tiers": {
                    "bronze": {
                        "tier": "bronze",
                        "model_name": "tfidf+logreg-test",
                        "model_version": "test",
                        "accuracy": 0.9,
                        "macro_f1": 0.9,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    yield target
    _remove_quietly(case)


@pytest.fixture()
def client(artifacts_dir: Path, monkeypatch):
    """TestClient wired to the temporary artifacts directory."""
    monkeypatch.setenv("NEWSCLASS_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("NEWSCLASS_DEFAULT_TIER", "bronze")
    monkeypatch.setenv("NEWSCLASS_ENVIRONMENT", "dev")

    from newsclass.config import get_settings, reset_settings_cache

    reset_settings_cache()
    settings = get_settings()
    # Force the fixture path even if a .env file overrides it.
    settings.artifacts_dir = artifacts_dir
    settings.default_tier = "bronze"

    from newsclass.app import create_app
    from fastapi.testclient import TestClient

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client

    reset_settings_cache()


@pytest.fixture()
def empty_client(monkeypatch):
    """TestClient with no trained models at all (degraded mode)."""
    case = _fresh_case_dir()
    monkeypatch.setenv("NEWSCLASS_ARTIFACTS_DIR", str(case / "empty"))
    monkeypatch.setenv("NEWSCLASS_DEFAULT_TIER", "gold")

    from newsclass.config import get_settings, reset_settings_cache

    reset_settings_cache()
    settings = get_settings()
    settings.artifacts_dir = case / "empty"
    settings.default_tier = "gold"

    from newsclass.app import create_app
    from fastapi.testclient import TestClient

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client

    reset_settings_cache()
    _remove_quietly(case)


@pytest.fixture()
def normalised_probabilities() -> np.ndarray:
    return np.array([[0.1, 0.2, 0.6, 0.1]])
