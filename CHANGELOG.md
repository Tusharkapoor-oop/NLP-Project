# Changelog

All notable changes to this project. Format follows Keep a Changelog and the
project adheres to Semantic Versioning.

## [1.0.0] - 2026-11-12

First production release. Covers the full lifecycle: data, training, packaging,
serving, observability and testing.

### Added

**Tiered model catalogue**
- `bronze` — TF-IDF + Logistic Regression. 0.83 MB, no neural network.
- `silver` — Word2Vec (CBOW) embeddings + BiLSTM. Own embeddings, trained from
  the training split only.
- `gold` — fine-tuned `distilbert-base-uncased`.
- A single `registry.json` records accuracy, macro-F1, size, latency and training
  time per tier, so the cost/quality trade-off is documented rather than implied.

**Serving**
- FastAPI application with versioned routes under `/v1`.
- `POST /v1/predict` and `POST /v1/predict/batch` with Pydantic validation.
- Automatic, logged fallback when the requested tier is unavailable.
- Separate liveness (`/healthz`) and readiness (`/readyz`) probes.
- Request IDs and `x-response-time-ms` on every response.

**Observability**
- Prometheus metrics: request counts per tier and class, latency histograms,
  confidence distribution (drift proxy), fallback counter.
- Pre-provisioned Prometheus + Grafana stack in `docker/`.

**Operations**
- Multi-stage, non-root Docker image with a baked-in healthcheck.
- `docker compose` stack for API + Prometheus + Grafana.
- `scripts/evaluate.py` with an optional accuracy gate for release checks.
- `scripts/smoke_test.py` for post-deploy verification.

**Engineering**
- 73 automated tests covering the API contract, routing, fallback, config
  validation and preprocessing.
- Typed configuration via `pydantic-settings`; nothing hard-coded.
- Shared preprocessing module to eliminate training/serving skew.

**Notebooks**
- `Word2Vec_LSTM_NLP_Classification.ipynb` — classical pipeline (AG News).
- `Transformer_DistilBERT_NLP_Classification.ipynb` — Transformer pipeline (BBC News).
- Both fully executed with saved outputs.

### Fixed

- FastAPI dependency resolution broke under `from __future__ import annotations`
  when dependencies were defined inside the app factory: FastAPI resolved the
  string annotations in module globals, failed to find them, and treated the
  parameter as a required query parameter (HTTP 422 on every route). Dependencies
  are now module-level.
- Training/serving skew in the bronze tier: the model was fitted on raw text but
  served normalised text. Both paths now use the same `normalize_text`.
- The silver tier crashed when NLTK WordNet data was absent. Lemmatization now
  degrades to a no-op instead of raising.

### Known limitations

- `gold` truncates documents to 128 sub-word tokens. On BBC News this covers only
  the article lead, which is measured and stated explicitly in the notebook.
- The tiers are evaluated on a single split with a fixed seed. Run-to-run
  variation is around ±0.5 accuracy points, so differences smaller than that
  should not be treated as meaningful.
- No authentication or rate limiting. A public deployment must add an API gateway
  or the `slowapi` middleware before exposing it to untrusted traffic.
