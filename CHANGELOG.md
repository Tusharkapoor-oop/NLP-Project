# Changelog

All notable changes to this project. Format follows Keep a Changelog and the
project adheres to Semantic Versioning.

## [2.0.0] - 2026-11-17

This release closes the evaluation gaps: the project no longer reports point
estimates as if they were facts. It also fixes the most serious defect found in
the project's history, and the fix **changes the headline numbers**.

### Fixed — data leakage that inflated the gold tier

**Symptom.** The gold tier's accuracy was reported as 0.9180 in the README while
the registry said 0.9067 and the accuracy audit said 0.9240. Four different files
disagreed about the same model.

**Cause.** `load_split(rows=N)` partitioned whatever range it was handed, so the
split depended on how many rows a caller asked for:

* `train_gold` trained on `load_split(rows=6000)` → 4,800 train / 600 test
* every evaluation script used `load_split(rows=20000)` → 16,000 train / 2,000 test

The two calls produce **different partitions of overlapping source rows**. Gold
was therefore trained on one split and scored on another, and 464 of its own
training documents (23.2% of the 2,000-document test set) reappeared at
evaluation. Gold scored 95.7% on those memorised documents and 90.2% on the rest.

**Fix.** `load_split` now computes a single canonical partition from a fixed
universe. A new `train_rows` argument subsamples **only the training portion**,
so `load_split(train_rows=6000)` and `load_split()` return byte-identical
validation and test sets. Gold was retrained from scratch on the clean split.

**Impact, stated plainly:**

| | before | after |
|---|---|---|
| gold accuracy | 0.9180 (leaked) | **0.9035** (clean) |
| leaked test documents | 464 (23.2%) | **0** |
| gold vs bronze significance | claimed best | **p = 0.27 — not significant** |
| gold vs silver significance | claimed best | **p = 0.07 — not significant** |

`tests/test_data_integrity.py` (12 tests) now fails if any of these reappear: a
test set that moves when training rows change, train/test or val/test overlap,
the gold training subset touching the test set, or lost stratification.

### Fixed — training/serving skew in the gold tier's truncation window

`train_gold` fine-tuned at `max_length=64` while `GoldModel` served at
`max_length=96`. The model received sequences longer than anything it saw during
fine-tuning, and nothing errored. Training now writes `serving.json` into the
artifact recording the exact window, and `GoldModel` reads it; the constructor
still accepts an explicit override. The existing artifact was backfilled.

### Added — the evaluation scripts that were missing

- **`scripts/significance.py`** — bootstrap confidence intervals (percentile,
  default 2,000 resamples) for accuracy and macro-F1, Cohen's κ and MCC per tier,
  and **McNemar's test on every tier pair**. Uses the exact binomial test below 25
  discordant pairs and the continuity-corrected χ² above it. `--require-order`
  turns "the docs claim a ladder" into a gate. Bonferroni-corrects across pairs.
- **`scripts/benchmark.py`** — p50/p90/p95/p99 latency, **concurrent throughput**
  (thread pool, so the number means something for a service), **peak RSS** sampled
  on a background thread, and **cold start measured in a fresh subprocess** (once
  a tier is resident, reloading it is a cache hit and would report near-zero).
  `--max-p99-ms` provides a latency budget.
- **`scripts/seed_variance.py`** — retrains across seeds and reports mean ± std
  and range. `--vary-split` optionally re-draws the partition. It distinguishes
  *deterministic* tiers (TF-IDF + lbfgs returns identical scores across seeds,
  which says nothing about pipeline variance) from stochastic ones, and says so
  rather than reporting "0.0000 variance" as reassurance.
- **`scripts/error_taxonomy.py`** — categorises every misclassification
  (`long_document`, `near_miss_boundary`, `confident_error`,
  `rare_token_overlap`, `other`) using signals available without human labels, so
  the categories are reproducible. Also slices accuracy by document-length
  tercile, by class, and by headline- vs article-shape. Correctly treats bronze as
  having **no** token window, so truncation is never blamed for a TF-IDF error.
- **`scripts/cost_model.py`** — derives USD per 1,000 requests from measured
  throughput and peak memory, with the pricing assumptions printed and the
  exclusions named. `--apply` writes `cost_per_1k_requests` into the registry,
  filling a column that had been `null` since 1.0.0.
- `scripts/run_pipeline.py` gained stages for the above (`--no-analysis`,
  `--resamples`, `--require-order`).

### Measured results (all reproducible from `artifacts/`)

**Accuracy, with 95% bootstrap confidence intervals, on the canonical
2,000-document test set:**

| Tier | Accuracy | 95% CI | Macro-F1 | κ | MCC |
|---|---|---|---|---|---|
| bronze | 0.8960 | [0.882, 0.909] | 0.8954 | 0.8612 | 0.8613 |
| silver | 0.8915 | [0.878, 0.906] | 0.8911 | 0.8553 | 0.8553 |
| gold | 0.9035 | [0.890, 0.917] | 0.9027 | 0.8713 | 0.8714 |

**No tier pair differs significantly** (McNemar with Bonferroni α=0.0167):
gold–silver p=0.066, gold–bronze p=0.270, silver–bronze p=0.402.

**Operational profile** (8 threads, `artifacts/benchmark_report.json`):

| Tier | p50 | p99 | Throughput | Peak RSS | Cold start | USD / 1k |
|---|---|---|---|---|---|---|
| bronze | 0.64 ms | 1.83 ms | 11,316 docs/s | 1,117 MB | 4.5 s | $0.00001 |
| silver | 2.48 ms | 9.99 ms | 1,571 docs/s | 1,141 MB | 3.4 s | $0.00006 |
| gold | 67.99 ms | 103.61 ms | 15 docs/s | 1,267 MB | 14.4 s | $0.00608 |

Gold costs **760× bronze** and is **759× slower** for an accuracy difference that
is not statistically significant. On this evidence bronze is the correct default.

**Run-to-run variance** (`--vary-split`, 3 seeds): bronze accuracy std 0.0038,
range 0.0075 — which validates and slightly widens the previous "±0.5 points"
caveat, now measured rather than asserted.

**Failure taxonomy** (gold, 1,000 documents, 95 errors): 61.1%
`near_miss_boundary` (the true class is the model's second choice), 23.2%
`long_document`, 12.6% `confident_error`, 3.2% `other`. The length slice confirms
the truncation cost: gold scores 0.938 on medium-length documents but 0.878 on the
longest tercile, and 0.917 on headline-shaped input against 0.870 on
article-shaped.

### Changed

- The README tier section was rewritten around the confidence intervals, the
  paired tests and the cost table, replacing claims the data does not support.
  The 2.67× training-data handicap on gold (6,000 rows against 16,000) is stated
  explicitly, because it is the reason the comparison cannot be read as
  "Transformers do not help".
- `.gitignore` no longer double-lists `pytest-of-*/`. Documentation screenshots in
  `docs/assets/` remain tracked; a blanket `*.png` rule from an earlier draft was
  removed because it would have silently frozen those images.

## [1.1.1] - 2026-11-16

### Added

- **`NEWSCLASS_PRELOAD_TIERS`** — choose which tiers are loaded into memory at
  boot, in addition to the default one. Only the default tier was warmed before,
  which kept boot fast but meant the first request for an unwarmed tier paid the
  whole lazy-load cost. Measured on this CPU: **gold takes 38.9 s to load**, so a
  first request for gold exceeded the smoke test's 30 s client timeout and failed
  intermittently depending on which tier was queried first. With
  `PRELOAD_TIERS=bronze,silver,gold` the worst first-request time drops to
  **0.07 s**. Each preload is individually guarded, so requesting a tier whose
  artifacts are absent logs a warning and continues rather than breaking boot.
  Unknown tier names are rejected at startup, because a typo would silently never
  warm a tier — the exact failure the setting exists to prevent.
- `scripts/smoke_test.py --timeout` (default 120 s, was a hard-coded 30 s). A
  tight timeout reports a false failure for a slow-but-healthy service, which is
  what happened above.
- `docker-compose.yml` and CI now set `PRELOAD_TIERS`, and `.env.example`
  documents it with the measured cost.

### Fixed

- **Vendored JavaScript that was downloaded but never called.** GSAP **Flip** and
  **Observer** were shipped and loaded on every page visit while nothing in the
  UI referenced them — 34 KB and two HTTP requests of pure page weight, plus a
  header comment and a `NOTICE.md` that listed them as dependencies. Both files
  and both `<script>` tags are removed. This is exactly the kind of defect a
  reviewer finds by opening DevTools, so it is now enforced rather than trusted:
  `tests/test_ui_assets.py` strips comments from the UI source and asserts that
  every loaded script's global actually appears in code, and that the removed
  files stay removed (404) and stay out of the markup.
- **Two writers on one CSS property.** The method rail (`#method-progress`) was
  animated by an `IntersectionObserver` writing `style.width` per active step
  *and* by a new `ScrollTrigger` scrub on the same property, which would make the
  bar jitter between the two values. Ownership is now exclusive: the rail is
  scroll-driven when ScrollTrigger is available, step-driven otherwise.

### Added

- **Real scroll-driven motion**, replacing the previously unused ScrollTrigger
  dependency: a scrubbed hero/canvas parallax, staggered card deal-ins for the
  static grids, a scroll-linked method rail, and a one-time `ScrollTrigger.refresh()`
  after `load` so late layout changes do not leave stale trigger positions.
  Everything is guarded by capability checks and `prefers-reduced-motion`, so the
  page is fully readable with scripts blocked.
- Card deal-in for the model registry grid runs inside `renderDesks()`, i.e.
  *after* the cards exist. Animating `#desks-grid` at init would have targeted
  the loading skeletons that the fetch then replaces.

### Changed

- `.gitignore` no longer double-lists `pytest-of-*/`. The documentation
  screenshots in `docs/assets/` remain tracked (the README embeds them); an
  earlier draft of this release added a blanket `*.png` rule that would have
  silently stopped those documented images from ever being updated, so it was
  removed rather than kept. A Model Lab screenshot was added at
  `docs/assets/lab.png` and is embedded in the README.

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
- `GET /lab` serves a standalone Model Lab: every tier runs independently on the
  same copy, each with its own label, full probability distribution, server and
  round-trip latency, explanation and a named fallback when one answered instead.
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
- 101 automated tests, including the explanation, feedback, insights, validation,
  Model Lab and shipped-UI contracts; the smoke test now drives every endpoint.

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
