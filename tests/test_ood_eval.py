"""Tests for the out-of-distribution evaluation harness.

The scoring helpers are pure, so they are tested directly. The network paths are
exercised against a stubbed opener so a broken feed cannot be mistaken for a
broken model, and the tier-mismatch guard is tested because silently scoring a
fallback tier is the one failure that would make the numbers a lie.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ood_eval  # noqa: E402

LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def rss(items: list[tuple[str, str]]) -> bytes:
    body = "".join(
        f"<item><title>{title}</title><description>{desc}</description></item>"
        for title, desc in items
    )
    return f"<rss><channel>{body}</channel></rss>".encode()


class FakeOpener:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.requested: list[str] = []

    def __call__(self, request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else request
        self.requested.append(url)
        return FakeResponse(self.payloads.get(url, rss([])))


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc) -> bool:
        return False


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------
def test_clean_strips_markup_and_unescapes_entities():
    assert ood_eval.clean("<p>Hello &amp; <b>world</b></p>") == "Hello & world"


def test_clean_collapses_whitespace_and_trims():
    assert ood_eval.clean("  a\n\n b \t c  ") == "a b c"


def test_clean_of_empty_text_is_empty():
    assert ood_eval.clean("") == ""
    assert ood_eval.clean("<br/>") == ""


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------
def test_fetch_labels_by_feed_dedupes_and_filters_short_items(monkeypatch):
    world = rss(
        [
            ("A long enough world headline about elections today", "with a description"),
            ("A long enough world headline about elections today", "duplicate title"),
            ("Short", ""),
            ("Another sufficiently long world headline for the test", ""),
        ]
    )
    opener = FakeOpener({ood_eval.FEEDS["World"]: world})
    monkeypatch.setattr(ood_eval.urllib.request, "urlopen", opener)

    docs = ood_eval.fetch(per_feed=45)

    assert docs[0]["gold"] == "World"
    assert len([d for d in docs if d["gold"] == "World"]) == 2  # duplicate + short dropped


def test_fetch_respects_the_per_feed_cap(monkeypatch):
    opener = FakeOpener(
        {
            url: rss(
                [(f"{label} headline {i} that is comfortably long enough", "d") for i in range(10)]
            )
            for label, url in ood_eval.FEEDS.items()
        }
    )
    monkeypatch.setattr(ood_eval.urllib.request, "urlopen", opener)

    docs = ood_eval.fetch(per_feed=3)

    assert len(docs) == 3 * len(ood_eval.FEEDS)
    assert set(d["gold"] for d in docs) == set(ood_eval.FEEDS)


def test_fetch_dedupes_the_same_story_across_sections(monkeypatch):
    """Guardian headlines appear in several feeds; each document is scored once."""
    shared = [("The same story appears in every section feed today", "d")]
    opener = FakeOpener({url: rss(shared) for url in ood_eval.FEEDS.values()})
    monkeypatch.setattr(ood_eval.urllib.request, "urlopen", opener)

    docs = ood_eval.fetch(per_feed=5)

    assert len(docs) == 1
    assert docs[0]["gold"] == "World"  # first feed wins


def test_fetch_truncates_long_documents(monkeypatch):
    opener = FakeOpener({ood_eval.FEEDS["World"]: rss([("x" * 5000, "y" * 5000)])})
    monkeypatch.setattr(ood_eval.urllib.request, "urlopen", opener)

    docs = ood_eval.fetch(per_feed=1)

    assert len(docs[0]["text"]) == 1000


def test_fetch_survives_an_empty_feed(monkeypatch):
    opener = FakeOpener({})
    monkeypatch.setattr(ood_eval.urllib.request, "urlopen", opener)
    assert ood_eval.fetch(per_feed=5) == []


# ---------------------------------------------------------------------------
# predict / post
# ---------------------------------------------------------------------------
def prediction(label: str = "World", confidence: float = 0.9) -> dict:
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {"World": confidence, "Sports": 0.0, "Business": 0.0, "Sci/Tech": 0.0},
    }


def fake_post_factory(served: str = "bronze", batch_sizes: list[int] | None = None):
    def fake_post(path, payload, timeout=600, base_url=None):
        assert path == "/v1/predict/batch"
        if batch_sizes is not None:
            batch_sizes.append(len(payload["texts"]))
        return {
            "tier": served,
            "predictions": [prediction() for _ in payload["texts"]],
        }

    return fake_post


def test_predict_batches_and_keeps_the_gold_label(monkeypatch):
    sizes: list[int] = []
    monkeypatch.setattr(ood_eval, "BATCH", 2)
    monkeypatch.setattr(ood_eval, "post", fake_post_factory(batch_sizes=sizes))
    docs = [{"text": f"doc {i}", "gold": "World"} for i in range(5)]

    rows = ood_eval.predict(docs, "bronze")

    assert sizes == [2, 2, 1]
    assert len(rows) == 5
    assert all(r["gold"] == "World" for r in rows)
    assert all(r["label"] == "World" for r in rows)


def test_predict_refuses_a_fallback_tier(monkeypatch):
    monkeypatch.setattr(ood_eval, "post", fake_post_factory(served="bronze"))

    with pytest.raises(SystemExit, match="requested tier='gold'"):
        ood_eval.predict([{"text": "doc", "gold": "World"}], "gold")


def test_predict_rejects_a_prediction_count_mismatch(monkeypatch):
    def short_post(path, payload, timeout=600, base_url=None):
        return {"tier": "bronze", "predictions": []}

    monkeypatch.setattr(ood_eval, "post", short_post)

    with pytest.raises(ValueError):
        ood_eval.predict([{"text": "doc", "gold": "World"}], "bronze")


def test_post_sends_json_to_the_requested_base_url(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(json.dumps({"tier": "bronze", "predictions": []}).encode())

    monkeypatch.setattr(ood_eval.urllib.request, "urlopen", fake_urlopen)

    result = ood_eval.post(
        "/v1/predict/batch", {"texts": ["x"], "tier": "bronze"}, base_url="http://box:9"
    )

    assert captured["url"] == "http://box:9/v1/predict/batch"
    assert captured["method"] == "POST"
    assert captured["body"]["tier"] == "bronze"
    assert result["tier"] == "bronze"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_wilson_handles_zero_rows():
    assert ood_eval.wilson(0, 0) == (0.0, 0.0)


def test_wilson_stays_inside_the_unit_interval():
    low, high = ood_eval.wilson(0, 5)
    assert low == 0.0 and 0.0 < high < 1.0
    low, high = ood_eval.wilson(5, 5)
    assert 0.0 < low < 1.0 and high == 1.0


def test_prf_of_an_absent_label_is_all_zero():
    rows = [{"gold": "World", "label": "World"}]
    assert ood_eval.prf(rows, "Sports") == (0.0, 0.0, 0.0, 0)


def test_prf_for_a_split_prediction():
    rows = [
        {"gold": "World", "label": "World"},
        {"gold": "Business", "label": "World"},
    ]
    precision, recall, f1, support = ood_eval.prf(rows, "World")
    assert (precision, recall, support) == (0.5, 1.0, 1)
    assert f1 == pytest.approx(2 / 3)


def test_report_of_an_empty_set_does_not_divide_by_zero(capsys):
    result = ood_eval.report("empty", [])

    assert result["n"] == 0
    assert result["accuracy"] == 0.0
    assert result["ci95"] == [0.0, 0.0]
    assert result["macro_f1"] == 0.0
    assert result["errors"] == []
    assert "n=0" in capsys.readouterr().out


def test_report_summarises_confidences_and_errors(capsys):
    rows = [
        {"gold": "World", "label": "World", "confidence": 0.95, "text": "a"},
        {"gold": "Sports", "label": "World", "confidence": 0.55, "text": "b"},
    ]

    result = ood_eval.report("bronze", rows)

    assert result["n"] == 2
    assert result["accuracy"] == 0.5
    assert result["mean_conf_correct"] == pytest.approx(0.95)
    assert result["mean_conf_wrong"] == pytest.approx(0.55)
    assert result["errors"] == [{"gold": "Sports", "pred": "World", "conf": 0.55, "text": "b"}]
    assert set(result["per_class"]) == set(LABELS)
    out = capsys.readouterr().out
    assert "=== bronze ===" in out
    assert "confusion" in out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def test_main_creates_its_scratch_directory_and_writes_results(monkeypatch, tmp_path: Path, capsys):
    """A fresh clone has no .pytest_runs; the first run must create it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ood_eval, "fetch", lambda per_feed=45: [{"text": "doc", "gold": "World"}])
    monkeypatch.setattr(
        ood_eval,
        "predict",
        lambda docs, tier, base_url=None: [
            {"text": "doc", "gold": "World", "label": "World", "confidence": 0.9}
        ],
    )

    ood_eval.main()

    assert (tmp_path / ".pytest_runs" / "ood_docs.json").exists()
    payload = json.loads((tmp_path / ".pytest_runs" / "fresh_ood_results.json").read_text())
    assert set(payload["tiers"]) == {"bronze", "silver", "gold"}
    assert "=== summary ===" in capsys.readouterr().out


def test_main_reuses_the_cache_without_fetching(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".pytest_runs" / "ood_docs.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps([{"text": "doc", "gold": "World"}]), encoding="utf-8")

    def explode(per_feed=45):
        raise AssertionError("fetch must not be called when the cache exists")

    monkeypatch.setattr(ood_eval, "fetch", explode)
    monkeypatch.setattr(
        ood_eval,
        "predict",
        lambda docs, tier, base_url=None: [
            {"text": "doc", "gold": "World", "label": "World", "confidence": 0.9}
        ],
    )

    ood_eval.main()
