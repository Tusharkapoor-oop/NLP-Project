# Changelog

All notable changes to this project. Format follows Keep a Changelog and the
project adheres to Semantic Versioning.

## [1.1.0] - 2026-11-15

A review-ready release: the service gains a real interface, per-token
explanations, a human-feedback loop, live traffic signals, and an accuracy audit
that states its own calibration and failure modes.

### Added

**Wire Desk interface**
- `GET /` now serves a self-contained, offline-capable newsroom UI at
  `src/newsclass/static/` (Three.js scene, GSAP timelines and ScrollTrigger,
  Motion micro-interactions; all vendored locally, no CDN).
- Live tier catalogue, single-story filing via `/v1/explain`, side-by-side desk
  comparison, a bulletin, and an in-page correction form backed by
  `/v1/feedback`.
- Live signals panel fed by `/v1/insights`: label mix, p50/p95 latency,
  confidence drift, tier usage and recent volume.
- Defensive behaviour: reduced-motion support, mobile layout, escaped user text,
  request timeouts, a failure toast, and a preloader that always dismisses.

**Explainability**
- `POST /v1/explain` adds `explanation_method`, `explanation_context` and a ranked
  `tokens` list. Bronze uses exact coefficient × TF-IDF contributions; silver and
  gold use gradient×input saliency. Failed explanations degrade to `unavailable`.

**Human feedback**
- `POST /v1/feedback` writes an append-only JSONL correction with the text, human
  label, machine label, confidence and tier; `GET /v1/feedback` returns recent
  entries plus an agreement summary.

**Live insights**
- `GET /v1/insights` exposes a bounded in-process window of prediction traffic
  (label mix, confidence, p50/p95 latency, fallbacks, per-label series).

**Accuracy audit**
- `scripts/accuracy_report.py` adds per-class precision/recall/F1, a confusion
  matrix, expected calibration error, latency and a 24-item curated probe set,
  with configurable gates. Current stratified audit (1,000 documents): gold
  0.9240 / silver 0.8910 / bronze 0.8990 accuracy; probes 24/24, 22/24 and 21/24.

**Pipeline**
- `scripts/run_pipeline.py`: idempotent preflight → train → registry-load → live
  evaluation gate → optional serve, with a machine-readable report. Child
  processes receive `PYTHONPATH=<repo>/src`, so the pipeline works without a
  package install.

**Verification**
- 97 automated tests, including the explanation, feedback, insights, validation
  and shipped-UI contracts; the smoke test now drives every endpoint.

### Fixed

- The serving stack could fail to construct at all when an old FastAPI met a
  current Starlette (`Router.__init__() got an unexpected keyword argument
  'on_startup'`). Requirements now pin a compatible floor
  (`fastapi>=0.163`, `starlette>=1.0,<2`).
- `/v1/predict/batch` unpacked four of the five values returned by the routing
  helper, turning every batch request into a 500. Batch traffic is now also
  recorded in metrics, insights and the Prometheus histogram.
- Validation errors echoed the entire rejected document (up to 20,000
  characters) in the 422 body. They now report the field and the reason only.
- The feedback list route contained unreachable duplicate code.
- The UI no longer sends the current textarea contents when correcting an older
  verdict; it sends the text that produced that verdict.
- The signals panel printed the API's UTC timestamp while the filing card printed
  local time. Both now render on the reader's clock.
- The bulletin rounded confidence to a whole percent while the verdict and the
  desk comparison showed one decimal; all three now agree.

### Verification

- `tests/test_ui_assets.py` guards the offline promise: no CDN reference may
  return, every vendored script must be served by the app, and a missing asset
  must be a clean 404.
- The Playwright smoke test now fails if any comparison desk reports a failure or
  the insights panel cannot reach the API, and captures the comparison panel
  in view rather than below the fold.

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

**Robustness**

- Test suite no longer depends on pytest's `tmp_path` machinery. Its numbered-dir
  factory garbage-collects scratch trees at fixture setup, which restricted
  environments (CI sandboxes, locked-down lab machines) refuse — every
  fixture-based test then died with a misleading `Access is denied` before the
  test code even ran. Fixtures now create scratch directories directly inside
  the project and clean up on teardown.
- `scripts/run_tests.py` added: a thin wrapper giving each run a fresh scratch
  directory and passing any pytest flags through. CI uses it too.

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

- `gold` truncates documents to 64 sub-word tokens (`max_length=64`, recorded in
  `artifacts/gold/metrics.json`). Long articles are represented by their lead;
  this is measured and stated explicitly rather than hidden.
- The tiers are evaluated on a single split with a fixed seed. Run-to-run
  variation is around ±0.5 accuracy points, so differences smaller than that
  should not be treated as meaningful.
- No authentication or rate limiting. A public deployment must add an API gateway
  or the `slowapi` middleware before exposing it to untrusted traffic.
