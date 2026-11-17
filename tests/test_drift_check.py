"""Tests for the scheduled drift check.

Everything here targets the unattended-run contract: exit 0 healthy, 1 alert,
2 not judged, no traceback for any input, and an alert that cannot be lost
because the webhook failed.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import drift_check  # noqa: E402


def make_rows(correct: int, wrong: int) -> list[dict]:
    return [{"label": "World", "gold": "World"} for _ in range(correct)] + [
        {"label": "Sports", "gold": "World"} for _ in range(wrong)
    ]


def make_docs(n: int = 60) -> list[dict]:
    return [{"text": f"headline number {i}", "gold": "World"} for i in range(n)]


def run_main(monkeypatch, tmp_path: Path, argv: list[str], predictions: dict[str, list[dict]]):
    """Run main() with the network and the real artifacts directory taken out."""
    # Never read the repository baseline: a test must not depend on (or be
    # rescued by) the real per-tier floors in artifacts/.
    if "--baseline-file" not in argv:
        argv = [*argv, "--baseline-file", str(tmp_path / "baseline.json")]
    monkeypatch.setattr(sys, "argv", ["drift_check.py", *argv])
    monkeypatch.setattr(drift_check, "fetch", lambda per_feed=45: make_docs())
    monkeypatch.setattr(drift_check, "predict", lambda docs, tier, base_url=None: predictions[tier])
    return drift_check.main()


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def test_evaluate_with_no_rows_does_not_judge():
    result = drift_check.evaluate([], None, 0.65, 0.10)
    assert result["status"] == "no_data"
    assert result["n"] == 0
    assert result["accuracy"] == 0.0
    assert result["ci95"] == [0.0, 0.0]
    assert result["drop"] is None
    assert result["reasons"] == []


def test_evaluate_healthy_against_a_baseline():
    result = drift_check.evaluate(make_rows(50, 0), 0.90, 0.65, 0.10)
    assert result["status"] == "ok"
    assert result["accuracy"] == 1.0
    assert result["drop"] == pytest.approx(-0.1)
    assert result["reasons"] == []


def test_evaluate_alert_on_absolute_floor_only():
    result = drift_check.evaluate(make_rows(30, 20), None, 0.65, 0.10)
    assert result["status"] == "alert"
    assert result["drop"] is None
    assert len(result["reasons"]) == 1
    assert "below floor" in result["reasons"][0]


def test_evaluate_alert_on_drop_only():
    result = drift_check.evaluate(make_rows(40, 10), 0.95, 0.50, 0.10)
    assert result["status"] == "alert"
    assert result["drop"] == pytest.approx(0.15)
    assert len(result["reasons"]) == 1
    assert "dropped" in result["reasons"][0]


def test_evaluate_drop_exactly_at_the_limit_is_not_an_alert():
    """The threshold is strict: a drop of exactly max_drop is acceptable."""
    result = drift_check.evaluate(make_rows(80, 20), 0.90, 0.50, 0.10)
    assert result["drop"] == pytest.approx(0.10)
    assert result["status"] == "ok"


def test_evaluate_accuracy_exactly_at_the_floor_is_not_an_alert():
    result = drift_check.evaluate(make_rows(65, 35), None, 0.65, 0.10)
    assert result["accuracy"] == pytest.approx(0.65)
    assert result["status"] == "ok"


def test_evaluate_reports_both_breaches():
    result = drift_check.evaluate(make_rows(10, 40), 0.90, 0.65, 0.10)
    assert result["status"] == "alert"
    assert len(result["reasons"]) == 2


def test_evaluate_zero_accuracy_with_zero_floor_is_ok():
    result = drift_check.evaluate(make_rows(0, 10), None, 0.0, 0.10)
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# baseline IO
# ---------------------------------------------------------------------------
def test_load_baseline_missing_file_is_empty(tmp_path: Path):
    assert drift_check.load_baseline(tmp_path / "absent.json") == {}


def test_load_baseline_invalid_json_warns_and_ignores(tmp_path: Path, capsys):
    path = tmp_path / "baseline.json"
    path.write_text("{broken", encoding="utf-8")
    assert drift_check.load_baseline(path) == {}
    assert "not valid JSON" in capsys.readouterr().out


def test_load_baseline_without_tiers_key_is_empty(tmp_path: Path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    assert drift_check.load_baseline(path) == {}


def test_write_baseline_keeps_floors_and_skips_empty_tiers(tmp_path: Path):
    path = tmp_path / "nested" / "baseline.json"
    results = {
        "bronze": {
            "accuracy": 0.72,
            "n": 149,
            "ci95": [0.64, 0.79],
            "min_accuracy": 0.62,
            "status": "ok",
        },
        "gold": {"accuracy": 0.0, "n": 0, "ci95": [0.0, 0.0], "min_accuracy": 0.63},
    }

    drift_check.write_baseline(path, results, "test")

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert set(saved["tiers"]) == {"bronze"}
    assert saved["tiers"]["bronze"]["min_accuracy"] == 0.62
    # A refreshed baseline must still load back with its own floor.
    assert drift_check.load_baseline(path)["bronze"]["min_accuracy"] == 0.62


# ---------------------------------------------------------------------------
# webhook
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status: int, body: bytes = b"ok") -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:  # pragma: no cover - not used, but part of the API
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc) -> bool:
        return False


def test_send_alert_posts_slack_compatible_json(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse(200)

    monkeypatch.setattr(drift_check.urllib.request, "urlopen", fake_urlopen)

    assert drift_check.send_alert("https://hooks.example/x", "alert text", {"t": {}}) is True
    assert captured["url"] == "https://hooks.example/x"
    assert captured["body"]["text"] == "alert text"
    assert captured["body"]["drift"] == {"t": {}}


def test_send_alert_reports_a_non_2xx_response(monkeypatch):
    monkeypatch.setattr(
        drift_check.urllib.request, "urlopen", lambda request, timeout=None: FakeResponse(500)
    )
    assert drift_check.send_alert("https://hooks.example/x", "alert", {}) is False


def test_send_alert_swallows_transport_errors(monkeypatch, capsys):
    def boom(request, timeout=None):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(drift_check.urllib.request, "urlopen", boom)

    assert drift_check.send_alert("https://hooks.example/x", "alert", {}) is False
    assert "could not deliver" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def test_main_not_judged_when_too_few_documents(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(drift_check, "fetch", lambda per_feed=45: make_docs(10))
    monkeypatch.setattr(drift_check, "predict", lambda docs, tier, base_url=None: make_rows(10, 0))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drift_check.py",
            "--tier",
            "bronze",
            "--json-out",
            str(tmp_path / "report.json"),
            "--history",
            str(tmp_path / "history.jsonl"),
        ],
    )
    report = tmp_path / "report.json"

    code = drift_check.main()

    assert code == 2
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["documents"] == 10


def test_main_alerts_and_calls_the_webhook(monkeypatch, tmp_path: Path, capsys):
    calls: list[str] = []
    monkeypatch.setenv("NEWSCLASS_ALERT_WEBHOOK", "https://hooks.example/x")
    monkeypatch.setattr(
        drift_check, "send_alert", lambda url, message, payload: calls.append(message) or True
    )
    predictions = {"bronze": make_rows(0, 60)}

    code = run_main(
        monkeypatch,
        tmp_path,
        [
            "--tier",
            "bronze",
            "--json-out",
            str(tmp_path / "report.json"),
            "--history",
            str(tmp_path / "history.jsonl"),
        ],
        predictions,
    )

    assert code == 1
    assert len(calls) == 1
    assert "drift alert" in calls[0]
    assert "bronze" in calls[0]
    assert "webhook delivered" in capsys.readouterr().out


def test_main_healthy_writes_report_and_appends_history(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    monkeypatch.delenv("NEWSCLASS_ALERT_WEBHOOK", raising=False)
    monkeypatch.setattr(
        drift_check, "send_alert", lambda url, message, payload: calls.append(message) or True
    )
    report = tmp_path / "report.json"
    history = tmp_path / "history.jsonl"
    argv = [
        "--tier",
        "bronze",
        "--json-out",
        str(report),
        "--history",
        str(history),
    ]
    predictions = {"bronze": make_rows(58, 2)}

    first = run_main(monkeypatch, tmp_path, argv, predictions)
    second = run_main(monkeypatch, tmp_path, argv, predictions)

    assert first == 0 and second == 0
    assert calls == []
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["tiers"]["bronze"]["status"] == "ok"
    assert payload["tiers"]["bronze"]["min_accuracy"] == 0.65
    assert len(history.read_text(encoding="utf-8").splitlines()) == 2


def test_main_uses_the_per_tier_floor_from_the_baseline(monkeypatch, tmp_path: Path):
    """A tier that is legitimately weak OOD must not alert on the global default."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tiers": {"silver": {"accuracy": 0.58, "n": 149, "min_accuracy": 0.50}},
            }
        ),
        encoding="utf-8",
    )
    predictions = {"silver": make_rows(33, 27)}  # 0.55: below 0.65, above its own 0.50

    code = run_main(
        monkeypatch,
        tmp_path,
        [
            "--tier",
            "silver",
            "--baseline-file",
            str(baseline),
            "--json-out",
            str(tmp_path / "report.json"),
            "--history",
            str(tmp_path / "history.jsonl"),
        ],
        predictions,
    )

    assert code == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["tiers"]["silver"]["min_accuracy"] == 0.50


def test_main_update_baseline_preserves_the_existing_floor(monkeypatch, tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tiers": {"bronze": {"accuracy": 0.70, "n": 149, "min_accuracy": 0.62}},
            }
        ),
        encoding="utf-8",
    )

    code = run_main(
        monkeypatch,
        tmp_path,
        [
            "--tier",
            "bronze",
            "--baseline-file",
            str(baseline),
            "--update-baseline",
            "--json-out",
            str(tmp_path / "report.json"),
            "--history",
            str(tmp_path / "history.jsonl"),
        ],
        {"bronze": make_rows(54, 6)},
    )

    assert code == 0
    saved = json.loads(baseline.read_text(encoding="utf-8"))
    assert saved["tiers"]["bronze"]["accuracy"] == pytest.approx(0.9)
    assert saved["tiers"]["bronze"]["min_accuracy"] == 0.62


def test_main_offline_without_cache_exits_cleanly(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(drift_check, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["drift_check.py", "--offline"])

    code = drift_check.main()

    assert code == 2
    assert "NOT JUDGED" in capsys.readouterr().out


def test_main_offline_uses_the_cached_document_set(monkeypatch, tmp_path: Path):
    cache = tmp_path / ".pytest_runs" / "ood_docs.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(make_docs(60)), encoding="utf-8")
    monkeypatch.setattr(drift_check, "ROOT", tmp_path)
    monkeypatch.setattr(drift_check, "predict", lambda docs, tier, base_url=None: make_rows(60, 0))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drift_check.py",
            "--offline",
            "--tier",
            "bronze",
            "--json-out",
            str(tmp_path / "report.json"),
            "--history",
            str(tmp_path / "history.jsonl"),
        ],
    )

    assert drift_check.main() == 0


def test_main_alerts_without_a_webhook_configured(monkeypatch, tmp_path: Path, capsys):
    """No webhook is not a silent failure: the exit code still carries the alert."""
    monkeypatch.delenv("NEWSCLASS_ALERT_WEBHOOK", raising=False)

    def exploding_send(url, message, payload):
        raise AssertionError("no webhook is configured; send_alert must not be called")

    monkeypatch.setattr(drift_check, "send_alert", exploding_send)

    code = run_main(
        monkeypatch,
        tmp_path,
        [
            "--tier",
            "bronze",
            "--json-out",
            str(tmp_path / "report.json"),
            "--history",
            str(tmp_path / "history.jsonl"),
        ],
        {"bronze": make_rows(10, 50)},
    )

    assert code == 1
    assert "no NEWSCLASS_ALERT_WEBHOOK set" in capsys.readouterr().out
