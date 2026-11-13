"""Smoke tests for FastAPI inference services (imports and routes)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "api"))

import pytest
from fastapi.testclient import TestClient


def test_word2vec_api_import():
    import word2vec_lstm_api as w2v_api
    assert w2v_api.app is not None


def test_word2vec_api_health_without_artifacts():
    import word2vec_lstm_api as w2v_api
    client = TestClient(w2v_api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "word2vec_bilstm_bbc"
    assert data["status"] in ("ready", "loading")


@pytest.mark.skipif(
    not (ROOT / "models" / "word2vec_lstm" / "deployment_metadata.json").exists(),
    reason="Word2Vec model artifacts not trained yet."
)
def test_word2vec_api_prediction_shape():
    import word2vec_lstm_api as w2v_api
    client = TestClient(w2v_api.app)
    resp = client.post("/predict", json={"texts": ["Stock market surges on quarterly earnings"], "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert len(data["predictions"][0]["top_predictions"]) == 3


def test_transformer_api_import():
    import transformer_api as bert_api
    assert bert_api.app is not None


def test_transformer_api_health_without_artifacts():
    import transformer_api as bert_api
    client = TestClient(bert_api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "distilbert-base-uncased"
    assert data["status"] in ("ready", "loading")


@pytest.mark.skipif(
    not (ROOT / "models" / "transformer" / "best" / "config.json").exists(),
    reason="Transformer model artifacts not trained yet."
)
def test_transformer_api_prediction_shape():
    import transformer_api as bert_api
    client = TestClient(bert_api.app)
    resp = client.post("/predict", json={"texts": ["The president signed a new trade agreement today"], "top_k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert len(data["predictions"][0]["top_predictions"]) == 2
