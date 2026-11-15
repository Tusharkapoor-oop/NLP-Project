"""API contract tests.

These assert the behaviour a production service must guarantee:
valid responses, correct status codes, graceful degradation, and clean input
validation. They run against a real (small) sklearn model, not a mock.
"""

from __future__ import annotations

import pytest


class TestProbes:
    def test_healthz_is_ok(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_readyz_reports_ready(self, client):
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_readyz_fails_without_models(self, empty_client):
        assert empty_client.get("/readyz").status_code == 503

    def test_healthz_ignores_missing_models(self, empty_client):
        # Liveness must not depend on models, otherwise k8s restarts a fine pod.
        assert empty_client.get("/healthz").status_code == 200

    def test_request_id_is_returned(self, client):
        response = client.get("/healthz")
        assert response.headers.get("x-request-id")
        assert response.headers.get("x-response-time-ms")

    def test_request_id_is_echoed(self, client):
        response = client.get("/healthz", headers={"x-request-id": "abc-123"})
        assert response.headers["x-request-id"] == "abc-123"


class TestTierCatalogue:
    def test_lists_all_three_tiers(self, client):
        response = client.get("/v1/tiers")
        assert response.status_code == 200
        tiers = response.json()
        assert [t["tier"] for t in tiers] == ["bronze", "silver", "gold"]

    def test_reports_accuracy_and_cost(self, client):
        tiers = {t["tier"]: t for t in client.get("/v1/tiers").json()}
        assert tiers["bronze"]["accuracy"] == pytest.approx(0.9)
        assert "description" in tiers["gold"]

    def test_model_info_endpoint(self, client):
        body = client.get("/v1/model").json()
        assert body["default_tier"] == "bronze"
        assert "labels" in body
        assert len(body["labels"]) == 4


class TestSinglePrediction:
    def test_returns_valid_schema(self, client):
        response = client.post("/v1/predict", json={"text": "stocks rise as markets gain"})
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {
            "label",
            "confidence",
            "scores",
            "tier",
            "model_version",
            "latency_ms",
        }

    def test_confidence_is_a_probability(self, client):
        body = client.post("/v1/predict", json={"text": "the team won the cup final"}).json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_scores_are_sorted_descending(self, client):
        body = client.post("/v1/predict", json={"text": "apple launches a new laptop"}).json()
        probabilities = [s["probability"] for s in body["scores"]]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_scores_sum_to_one(self, client):
        body = client.post("/v1/predict", json={"text": "the president met ministers"}).json()
        total = sum(s["probability"] for s in body["scores"])
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_returns_all_four_classes(self, client):
        body = client.post("/v1/predict", json={"text": "market update"}).json()
        assert len(body["scores"]) == 4

    def test_reports_serving_tier(self, client):
        body = client.post("/v1/predict", json={"text": "market update"}).json()
        assert body["tier"] == "bronze"

    def test_latency_is_reported(self, client):
        body = client.post("/v1/predict", json={"text": "market update"}).json()
        assert body["latency_ms"] >= 0

    def test_business_text_classifies_reasonably(self, client):
        # Not a hard accuracy claim, just that the pipeline is wired correctly.
        body = client.post(
            "/v1/predict", json={"text": "stocks rise as market gains on earnings"}
        ).json()
        assert body["label"] in {"Business", "World", "Sports", "Sci/Tech"}


class TestValidation:
    def test_rejects_empty_text(self, client):
        assert client.post("/v1/predict", json={"text": ""}).status_code == 422

    def test_rejects_whitespace_only(self, client):
        assert client.post("/v1/predict", json={"text": "   "}).status_code == 422

    def test_rejects_missing_field(self, client):
        assert client.post("/v1/predict", json={}).status_code == 422

    def test_rejects_unknown_tier(self, client):
        response = client.post("/v1/predict", json={"text": "hello", "tier": "platinum"})
        assert response.status_code == 422

    def test_rejects_oversized_text(self, client):
        huge = "word " * 60_000
        response = client.post("/v1/predict", json={"text": huge})
        assert response.status_code in (413, 422)

    def test_validation_error_does_not_echo_the_document(self, client):
        # A rejected body must not be copied back verbatim: the response would
        # be huge and would duplicate text that could be sensitive.
        secret = "internal-draft-" * 2_000
        response = client.post("/v1/predict", json={"text": secret})
        assert response.status_code == 422
        assert "internal-draft" not in response.text
        assert "string_too_long" in response.text


class TestBatchPrediction:
    def test_batch_returns_one_result_per_input(self, client):
        texts = ["stocks rise", "the team won", "new laptop launched"]
        response = client.post("/v1/predict/batch", json={"texts": texts})
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 3

    def test_batch_rejects_empty_list(self, client):
        assert client.post("/v1/predict/batch", json={"texts": []}).status_code == 422

    def test_batch_rejects_blank_item(self, client):
        assert client.post("/v1/predict/batch", json={"texts": ["ok", "  "]}).status_code == 422

    def test_batch_enforces_max_size(self, client):
        texts = [f"story number {i}" for i in range(300)]
        assert client.post("/v1/predict/batch", json={"texts": texts}).status_code == 413


class TestDegradation:
    def test_predict_without_models_returns_503(self, empty_client):
        response = empty_client.post("/v1/predict", json={"text": "hello world"})
        assert response.status_code == 503
        assert "detail" in response.json()

    def test_fallback_to_available_tier(self, client):
        # Ask for gold; only bronze exists, so it must fall back rather than 500.
        response = client.post("/v1/predict", json={"text": "market update", "tier": "gold"})
        assert response.status_code == 200
        assert response.json()["tier"] == "bronze"


class TestMetrics:
    def test_metrics_endpoint_exposes_prometheus_format(self, client):
        client.post("/v1/predict", json={"text": "market update"})
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "newsclass_predictions_total" in response.text
        assert "newsclass_prediction_latency_seconds" in response.text

    def test_confidence_histogram_is_recorded(self, client):
        client.post("/v1/predict", json={"text": "market update"})
        text = client.get("/metrics").text
        assert "newsclass_prediction_confidence" in text


class TestDocs:
    def test_openapi_is_served(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "/v1/predict" in response.json()["paths"]


class TestExplanation:
    def test_explain_returns_token_evidence(self, client):
        response = client.post(
            "/v1/explain", json={"text": "stocks rise as the market gains on earnings"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["label"]
        assert body["explanation_method"]
        assert body["explanation_method"] != "unavailable"
        assert isinstance(body["tokens"], list)
        assert body["tokens"], "the linear tier must attribute at least one token"
        assert "token" in body["tokens"][0]

    def test_tokens_are_ranked_by_absolute_contribution(self, client):
        body = client.post(
            "/v1/explain", json={"text": "apple launches a new laptop with a faster chip"}
        ).json()
        magnitudes = [abs(token["contribution"] or 0.0) for token in body["tokens"]]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_explain_rejects_unknown_tier(self, client):
        response = client.post("/v1/explain", json={"text": "hello world", "tier": "platinum"})
        assert response.status_code == 422

    def test_explain_rejects_blank_text(self, client):
        assert client.post("/v1/explain", json={"text": "   "}).status_code == 422


class TestFeedbackLoop:
    def test_records_and_lists_correction(self, client):
        payload = {
            "text": "the team won the cup final",
            "corrected_label": "Sports",
            "predicted_label": "World",
            "predicted_confidence": 0.41,
            "tier": "bronze",
            "reason": "test correction",
        }
        response = client.post("/v1/feedback", json=payload)
        assert response.status_code == 201
        entry = response.json()
        assert entry["corrected_label"] == "Sports"
        assert entry["predicted_label"] == "World"
        assert entry["id"]

        listed = client.get("/v1/feedback").json()
        assert len(listed["entries"]) == 1
        assert listed["entries"][0]["id"] == entry["id"]
        assert listed["summary"]["total"] == 1

    def test_rejects_an_unknown_label(self, client):
        payload = {
            "text": "some copy",
            "corrected_label": "Politics",
            "predicted_label": "World",
            "predicted_confidence": 0.5,
            "tier": "bronze",
        }
        assert client.post("/v1/feedback", json=payload).status_code == 422

    def test_agreement_is_tracked_separately_from_corrections(self, client):
        def send(predicted: str, corrected: str) -> None:
            response = client.post(
                "/v1/feedback",
                json={
                    "text": "the team won the cup final",
                    "corrected_label": corrected,
                    "predicted_label": predicted,
                    "predicted_confidence": 0.8,
                    "tier": "bronze",
                },
            )
            assert response.status_code == 201

        send("Sports", "Sports")  # a confirm
        send("World", "Sports")  # a fix
        summary = client.get("/v1/feedback").json()["summary"]
        assert summary["total"] == 2
        assert summary["agreement_with_human"] == pytest.approx(0.5)
        assert summary["by_tier"] == {"bronze": 2}

    def test_rejects_impossible_confidence(self, client):
        response = client.post(
            "/v1/feedback",
            json={
                "text": "some copy",
                "corrected_label": "World",
                "predicted_label": "Sports",
                "predicted_confidence": 1.4,
                "tier": "bronze",
            },
        )
        assert response.status_code == 422

    def test_limit_is_validated(self, client):
        assert client.get("/v1/feedback", params={"limit": 0}).status_code == 422


class TestInsights:
    def test_empty_before_any_traffic(self, client):
        body = client.get("/v1/insights").json()
        assert body["empty"] is True
        assert body["window"]["recent_events"] == 0
        assert body["label_mix"] == []

    def test_reflects_recent_traffic(self, client):
        client.post("/v1/predict", json={"text": "stocks rise as the market gains"})
        response = client.get("/v1/insights")
        assert response.status_code == 200
        body = response.json()
        assert body["empty"] is False
        assert body["window"]["recent_events"] >= 1
        assert body["label_mix"]
        assert "p50_ms" in body["latency"]

    def test_batch_calls_are_counted(self, client):
        client.post("/v1/predict/batch", json={"texts": ["stocks rise", "the cup final"]})
        body = client.get("/v1/insights").json()
        assert body["window"]["recent_events"] == 2

    def test_explanation_requests_are_counted(self, client):
        client.post("/v1/explain", json={"text": "stocks rise as the market gains"})
        body = client.get("/v1/insights").json()
        assert body["window"]["recent_events"] == 1


class TestUi:
    def test_root_serves_the_single_page_ui(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Wire Desk" in response.text

    def test_static_assets_are_served(self, client):
        response = client.get("/static/assets/app.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]
        assert "fileStory" in response.text
