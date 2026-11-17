"""Tests for adjudicated rescoring (reviewer ledger -> corrected accuracy)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import adjudicate  # noqa: E402

LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def entry(text: str, adjudicated: str, verdict: str = "corrected", note: str = "why") -> dict:
    return {
        "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "section": "Business",
        "gold_pred": "Business",
        "silver_pred": "World",
        "adjudicated": adjudicated,
        "verdict": verdict,
        "note": note,
        "id": "doc",
    }


def row(text: str, gold: str, label: str, confidence: float = 0.9) -> dict:
    return {"text": text, "gold": gold, "label": label, "confidence": confidence}


# ---------------------------------------------------------------------------
# ledger IO
# ---------------------------------------------------------------------------
def test_sha1_is_stable_for_unicode_text():
    assert adjudicate.sha1("café — 東京") == adjudicate.sha1("café — 東京")
    assert adjudicate.sha1("a") != adjudicate.sha1("b")


def test_load_ledger_missing_file_is_a_clean_exit(tmp_path: Path):
    with pytest.raises(SystemExit, match="missing adjudication ledger"):
        adjudicate.load_ledger(tmp_path / "absent.jsonl")


def test_load_ledger_skips_blank_lines_and_last_entry_wins(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    first = entry("same text", "World")
    second = entry("same text", "Sci/Tech")
    path.write_text(
        "\n".join([json.dumps(first), "", "   ", json.dumps(second)]) + "\n", encoding="utf-8"
    )

    ledger = adjudicate.load_ledger(path)

    assert len(ledger) == 1
    assert ledger[first["sha1"]]["adjudicated"] == "Sci/Tech"


def test_load_ledger_names_the_bad_line(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(entry("ok", "World")) + "\n{oops\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="ledger.jsonl:2"):
        adjudicate.load_ledger(path)


def test_load_ledger_rejects_an_entry_without_a_sha1(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps({"verdict": "upheld"}) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="ledger.jsonl:1"):
        adjudicate.load_ledger(path)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_wilson_handles_zero_rows():
    assert adjudicate.wilson(0, 0) == (0.0, 0.0)


def test_wilson_bounds_stay_inside_the_unit_interval():
    low, high = adjudicate.wilson(0, 10)
    assert low == 0.0
    assert 0.0 < high < 1.0

    low, high = adjudicate.wilson(10, 10)
    assert 0.0 < low < 1.0
    assert high == 1.0


def test_prf_of_an_absent_label_is_all_zero():
    rows = [{"truth": "World", "label": "World"}]
    assert adjudicate.prf(rows, "Sports") == (0.0, 0.0, 0.0, 0)


def test_prf_counts_a_constant_predictor():
    rows = [{"truth": "World", "label": "World"}, {"truth": "Sports", "label": "World"}]
    precision, recall, f1, support = adjudicate.prf(rows, "World")
    assert (precision, recall, support) == (0.5, 1.0, 1)
    assert f1 == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def test_score_without_a_ledger_falls_back_to_section_labels():
    rows = [row("a", "World", "World"), row("b", "Sports", "World")]
    result = adjudicate.score(rows, {})

    assert result["n"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == 0.5
    assert result["relabelled"] == 0
    assert result["covered_by_ledger"] == 0
    assert all(r["source"] == "section" for r in result["rows"])


def test_score_empty_input_is_not_a_division_by_zero():
    result = adjudicate.score([], {})
    assert result["n"] == 0
    assert result["accuracy"] == 0.0
    assert result["ci95"] == [0.0, 0.0]
    assert result["macro_f1"] == 0.0


def test_excluded_document_leaves_the_denominator():
    rows = [row("promo", "Business", "World"), row("real", "World", "World")]
    ledger = {adjudicate.sha1("promo"): entry("promo", "EXCLUDE", "excluded", "newsletter promo")}

    result = adjudicate.score(rows, ledger)

    assert result["n"] == 1
    assert result["excluded"] == 1
    assert result["accuracy"] == 1.0
    assert result["errors_excluded"] == 1
    assert result["excluded_items"][0]["reason"] == "newsletter promo"
    assert result["covered_by_ledger"] == 1


def test_a_corrected_section_label_removes_a_phantom_error():
    """Model said Sci/Tech, section said Business, reviewer agrees with the model."""
    rows = [row("gta vi feature", "Business", "Sci/Tech")]
    ledger = {adjudicate.sha1("gta vi feature"): entry("gta vi feature", "Sci/Tech")}

    result = adjudicate.score(rows, ledger)

    assert result["accuracy"] == 1.0
    assert result["published_errors"] == 1
    assert result["label_noise_removed"] == 1
    assert result["final_errors"] == 0
    assert result["errors_revealed"] == 0


def test_a_correction_away_from_the_model_reveals_a_hidden_error():
    """Model agreed with the wrong section label; the reviewer does not."""
    rows = [row("nvidia board", "Business", "Business")]
    ledger = {adjudicate.sha1("nvidia board"): entry("nvidia board", "Sci/Tech")}

    result = adjudicate.score(rows, ledger)

    assert result["accuracy"] == 0.0
    assert result["published_errors"] == 0
    assert result["label_noise_removed"] == 0
    assert result["errors_revealed"] == 1
    assert result["final_errors"] == 1


def test_a_correction_that_matches_neither_label_is_only_a_genuine_error():
    rows = [row("mixed story", "World", "Business")]
    ledger = {adjudicate.sha1("mixed story"): entry("mixed story", "Sci/Tech")}

    result = adjudicate.score(rows, ledger)

    assert result["accuracy"] == 0.0
    assert result["label_noise_removed"] == 0
    assert result["errors_revealed"] == 0
    assert result["final_errors"] == 1


def test_identical_texts_share_one_verdict():
    rows = [row("dup", "Business", "Sci/Tech"), row("dup", "Business", "Sci/Tech")]
    ledger = {adjudicate.sha1("dup"): entry("dup", "Sci/Tech")}

    result = adjudicate.score(rows, ledger)

    assert result["n"] == 2
    assert result["correct"] == 2
    assert result["relabelled"] == 2


def test_error_accounting_reconciles():
    rows = [
        row("kept-correct", "World", "World"),
        row("noise", "Business", "Sci/Tech"),
        row("revealed", "Business", "Business"),
        row("genuine", "World", "Sports"),
        row("excluded", "Business", "World"),
        row("both-wrong", "World", "Business"),
    ]
    ledger = {
        adjudicate.sha1("noise"): entry("noise", "Sci/Tech"),
        adjudicate.sha1("revealed"): entry("revealed", "Sci/Tech"),
        adjudicate.sha1("excluded"): entry("excluded", "EXCLUDE", "excluded"),
        adjudicate.sha1("both-wrong"): entry("both-wrong", "Sci/Tech"),
    }

    result = adjudicate.score(rows, ledger)

    assert (
        result["published_errors"]
        - result["label_noise_removed"]
        - result["errors_excluded"]
        + result["errors_revealed"]
        == result["final_errors"]
    )
    assert result["final_errors"] == len(result["rows"]) - result["correct"]
    assert result["n"] + result["excluded"] == len(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def write_results(path: Path, tiers: dict[str, list[dict]]) -> None:
    path.write_text(
        json.dumps(
            {"source": "test", "docs": 0, "tiers": {t: {"rows": rows} for t, rows in tiers.items()}}
        ),
        encoding="utf-8",
    )


def test_main_requires_a_results_file(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "argv", ["adjudicate.py", "--results", str(tmp_path / "absent.json")])
    with pytest.raises(SystemExit, match="missing results file"):
        adjudicate.main()


def test_main_requires_the_ledger(monkeypatch, tmp_path: Path):
    results = tmp_path / "results.json"
    write_results(results, {"gold": [row("a", "World", "World")]})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adjudicate.py",
            "--results",
            str(results),
            "--ledger",
            str(tmp_path / "absent.jsonl"),
        ],
    )
    with pytest.raises(SystemExit, match="missing adjudication ledger"):
        adjudicate.main()


def test_main_writes_a_report_without_the_row_payload(monkeypatch, tmp_path: Path, capsys):
    results = tmp_path / "results.json"
    ledger = tmp_path / "ledger.jsonl"
    out = tmp_path / "report.json"
    write_results(results, {"gold": [row("a", "World", "World"), row("b", "Business", "Sci/Tech")]})
    ledger.write_text(json.dumps(entry("b", "Sci/Tech")) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adjudicate.py",
            "--results",
            str(results),
            "--ledger",
            str(ledger),
            "--tier",
            "gold",
            "--json-out",
            str(out),
        ],
    )

    assert adjudicate.main() == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["tiers"]) == {"gold"}
    assert "rows" not in payload["tiers"]["gold"]
    assert payload["tiers"]["gold"]["accuracy"] == 1.0
    assert "as published" in capsys.readouterr().out


def test_running_as_a_script_exits_nonzero_with_a_readable_error(tmp_path: Path):
    """The __main__ guard, exercised the way a user runs it."""
    completed = subprocess.run(
        [sys.executable, "scripts/adjudicate.py", "--results", str(tmp_path / "absent.json")],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert completed.returncode == 1
    assert "missing results file" in completed.stderr
