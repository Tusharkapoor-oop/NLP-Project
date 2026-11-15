# News Topic Classification — Production NLP Service

An end-to-end, deployable NLP system that classifies news text into four topics
(**World, Sports, Business, Sci/Tech**). It serves three interchangeable model
tiers so accuracy can be traded against cost per request.

This is not a notebook that stops at `plt.show()`. It ships a containerised API,
observability, automated tests, an evaluation gate, and a documented model risk
policy.

```
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
│  Dataset    │──▶│  Training    │──▶│  Registry    │──▶│  FastAPI    │
│  AG News    │   │  3 tiers     │   │ registry.json│   │  /v1/predict│
│ (HF Hub)    │   │ bronze/silver│   │ + metrics    │   │  fallback   │
└─────────────┘   │ /gold        │   └──────────────┘   └──────┬──────┘
                  └──────────────┘                              │
                                                  ┌─────────────┴───────────┐
                                                  │  Prometheus + Grafana   │
                                                  └─────────────────────────┘
```

---

## Quick start (60 seconds)

```bash
# 1. install
pip install -r requirements-serve.txt

# 2. train the cheapest tier, verify it, gate it, and serve it
python scripts/run_pipeline.py --only bronze --serve
```

`run_pipeline.py` is the only command you need: it preflights the environment,
trains whatever is missing, re-loads every artifact through the real registry,
re-scores it on held-out data, refuses to serve a model below the macro-F1 gate,
and only then boots the API. It is idempotent, so running it twice costs nothing.

```bash
python scripts/run_pipeline.py --check          # verify only, touch nothing
python scripts/run_pipeline.py --only bronze    # one tier
python scripts/run_pipeline.py --serve --port 8010
```

Open **http://localhost:8000/** for the web UI, or **/docs** for the interactive
API reference. Then try `POST /v1/predict`:

```json
{ "text": "Apple unveils a chip that is twice as fast as the last one." }
```

```json
{
  "label": "Sci/Tech",
  "confidence": 0.93,
  "scores": [{ "label": "Sci/Tech", "probability": 0.93 }, "..."],
  "tier": "bronze",
  "model_version": "1.0.0",
  "latency_ms": 1.4
}
```

Full stack with metrics dashboards:

```bash
docker compose -f docker/docker-compose.yml up --build
# API   http://localhost:8000/docs
# Prom   http://localhost:9090
# Grafana http://localhost:3000
```

---

## The web UI ("Wire Desk")

`GET /` serves a self-contained newsroom interface, not a Swagger page.

![The Wire Desk hero](docs/assets/hero.png)

| What it demonstrates | How |
|---|---|
| The tier trade-off, made tangible | File one story and compare all three desks side by side; each row reports its own label and confidence |
| Model evidence, not just a label | **Explain** calls `/v1/explain` and shows per-token contributions (exact coefficient × TF-IDF for bronze, gradient×input for the neural tiers) |
| The feedback loop | **Send correction** writes to `/v1/feedback`; agreement and disagreement are both recorded as retraining data |
| Live operational signals | `/v1/insights` backs a bounded traffic window: label mix, p50/p95 latency, confidence drift, tier usage |
| It works offline | Three.js, GSAP and Motion are vendored under `static/vendor/`; the page makes no CDN requests |

![One copy, three desks](docs/assets/studio-compare.png)

The interface is deliberately defensive: if a vendor library or an API call
fails, the page falls back to readable static content and a toast instead of a
blank screen. `tests/test_ui_assets.py` enforces that no CDN reference sneaks
back in.

---

## The three tiers, and what each one is *for*

This is the core design decision. Rather than shipping one model, the system ships
a **quality/cost ladder**. The same request can be served at three different price
points, and the choice is an explicit, measurable trade-off.

| Tier | Model | Accuracy | Macro-F1 | Size | Latency / 100 docs | Train (CPU) | Use when |
|---|---|---|---|---|---|---|---|
| **Bronze** | TF-IDF + Logistic Regression | 0.8990 | 0.8989 | 0.83 MB | 11 ms | ~21 s | High volume, cost-sensitive, no PyTorch runtime needed |
| **Silver** | Word2Vec (CBOW) + BiLSTM | 0.8910 | 0.8910 | 9.2 MB | 1,145 ms | ~7 min | You want learned embeddings and sequence modelling on CPU |
| **Gold** | Fine-tuned DistilBERT | 0.9240 | 0.9238 | 268.6 MB | 18,669 ms | ~88 min | Accuracy matters more than latency or memory |

All figures come from `python scripts/accuracy_report.py` — a stratified
1,000-document audit of the held-out split (seed 42).
`artifacts/accuracy_report.json` holds the full output, including per-class scores,
confusion matrices and calibration. Three honest readings of this table are worth
stating out loud:

* **Gold buys ~2.5 accuracy points over bronze** at ~324× the size and ~1,600× the
  latency. Whether that trade is worth it is a business decision, which is the
  whole point of routing by tier.
* **Silver is not between its neighbours at all.** It is ~100× slower than bronze
  yet ~0.8 points *less* accurate on this dataset — sequence modelling pays off
  on longer, more nuanced documents, and AG News headlines are short and
  keyword-rich. Reporting that honestly is more useful than drawing a tidy ladder.
* **The tiers were not trained on identical row counts.** Bronze and silver used
  16,000 training / 2,000 test rows; the gold run used 4,800 / 600 to fit the CPU
  budget. That makes gold's number noisier, and it is stated here rather than
  hidden by a rounder number.

> **Actively measured numbers, not marketing.** `artifacts/registry.json` holds the
> training-time figures; `artifacts/accuracy_report.json` holds the fresh audit.
> Regenerate either with `python scripts/evaluate.py` or
> `python scripts/accuracy_report.py` — both re-score the held-out split.

### Why a tier ladder rather than one "best" model?

A strict reviewer will ask "why not just use the best model?" The honest answer is
that the best model is ~324× larger and ~248× slower to train, and on this dataset
it buys ~2.5 accuracy points. Whether that is worth it depends entirely on
traffic volume and the cost of a wrong answer:

- **Bulk classification of a million headlines** — bronze. The marginal accuracy is
  not worth 269 MB per replica times N replicas.
- **Moderating user-facing content** — gold. A wrong label is visible to a user.
- **A useful middle ground** — silver. Sequence-aware, still CPU-friendly.

The API reports which tier served every response, so this decision is auditable
after the fact rather than hidden inside a config file.

---

## What "production" means here (the checklist)

| Concern | How it is handled | Where |
|---|---|---|
| Config without redeploys | Typed settings from environment variables | `src/newsclass/config.py` |
| One-command pipeline | Preflight → train → verify → gate → serve, idempotent, non-zero on failure | `scripts/run_pipeline.py` |
| Training/serving skew | One shared preprocessing module, imported by both | `src/newsclass/text.py` |
| Graceful degradation | Falls back to the best available tier, logged and metriced | `src/newsclass/registry.py` |
| Liveness vs readiness | Separate `/healthz` and `/readyz`; liveness ignores models | `src/newsclass/app.py` |
| Metrics | Request counts, latency histograms, confidence drift, fallbacks | `src/newsclass/metrics.py` |
| Stateless containers | Models read from a mounted volume; no writes at runtime | `docker/Dockerfile` |
| Reproducible builds | Multi-stage image, pinned base, non-root user | `docker/Dockerfile` |
| Automated verification | 97 tests, CI matrix, accuracy gate, container smoke test | `tests/`, `.github/workflows/ci.yml` |
| Release gate | Refuses to ship an artifact below a macro-F1 threshold | `scripts/evaluate.py --gate` |
| Accuracy audit | Per-class scores, calibration error, curated probes; gated in CI | `scripts/accuracy_report.py` |
| Explainability | Native per-tier attributions (exact linear, gradient×input neural) | `src/newsclass/explain.py` |
| Human feedback | Append-only JSONL correction ledger with the machine's verdict | `src/newsclass/feedback.py` |
| Privacy in errors | 422 responses report the field and reason, never echo the document | `src/newsclass/app.py` |
| Offline UI | Vendored Three.js / GSAP / Motion; no CDN requests | `src/newsclass/static/` |
| Observability stack | Pre-provisioned Prometheus + Grafana | `docker/` |
| Governance | Model card, risk policy, changelog | `docs/`, `CHANGELOG.md` |

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | The Wire Desk UI (self-contained, offline-capable) |
| `GET` | `/healthz` | Liveness. 200 while the process runs. Never checks models. |
| `GET` | `/readyz` | Readiness. 503 when no tier can serve. |
| `GET` | `/v1/tiers` | Every tier with accuracy, size, latency and a human description. |
| `GET` | `/v1/model` | Active default tier, loaded tiers, label set. |
| `POST` | `/v1/predict` | Classify one document. |
| `POST` | `/v1/predict/batch` | Classify up to `max_batch_size` documents. |
| `POST` | `/v1/explain` | Classify **and** attribute the verdict to tokens, in one call. |
| `POST` | `/v1/feedback` | Record a human correction; returns the stored entry. |
| `GET` | `/v1/feedback` | Recent corrections plus an agreement summary. |
| `GET` | `/v1/insights` | Bounded in-process traffic window: label mix, latency, drift. |
| `GET` | `/metrics` | Prometheus exposition. |

Every response carries `x-request-id` and `x-response-time-ms`.

### Explainability

```bash
curl -X POST localhost:8000/v1/explain \
  -H 'Content-Type: application/json' \
  -d '{"text": "Apple unveils a chip that is twice as fast as the last one."}'
```

The response extends `PredictionResponse` with `explanation_method`,
`explanation_context` and a ranked `tokens` list. The method is native to the
tier and named in the payload, so the evidence is never mistaken for more than
it is: exact linear contributions for TF-IDF, gradient×input saliency for the
neural tiers. A failed explanation degrades to
`"explanation_method": "unavailable"` without failing the prediction.

### Feedback loop

A correction is training data, not a complaint box. Each entry stores the text,
the corrected label, the predicted label, its confidence and the serving tier,
so a review can replay exactly where the model and the human disagreed.
`GET /v1/feedback` returns both the entries and `agreement_with_human`, which is
1.0 when every correction matched the prediction.

### Choosing a tier per request

```bash
curl -X POST localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "The striker scored twice.", "tier": "gold"}'
```

If `gold` is not installed, the response is served by the best available tier and
`newsclass_tier_fallbacks_total` increments. The caller always gets an answer;
the degradation is visible in metrics, never silent.

---

## Training

```bash
python scripts/run_pipeline.py --only bronze   # preflight + train + verify + gate
python scripts/run_pipeline.py                 # every missing tier
python scripts/run_pipeline.py --force         # retrain everything
```

Under the hood each tier is still trained by `python -m newsclass.train --tier X`
(`bronze` ~21 s, `silver` ~7 min, `gold` ~88 min on this CPU). Each tier writes
its weights plus a `metrics.json`, and `registry.json` aggregates them. The
registry is what the API reads at boot.

### Datasets

| | Dataset | Source | Rows | Classes |
|---|---|---|---|---|
| Serving pipeline | **AG News** | `SetFit/ag_news` on Hugging Face | 20,000 used (120,000 available) | 4 |
| Notebook 2 | **BBC News** | `SetFit/bbc-news` on Hugging Face | 1,225 train / 1,000 test | 5 |

Both are public, versioned on the Hugging Face Hub, and downloaded at runtime —
no scraped URLs, no bundled data files.

---

## Evaluation

```bash
python scripts/evaluate.py --gate 0.85          # release gate, all tiers
python scripts/accuracy_report.py               # per-class, calibration, probes
```

```
gold    acc=0.9240 macro_f1=0.9238 conf=0.940 ECE=0.028 18669 ms/100 docs
silver  acc=0.8910 macro_f1=0.8910 conf=0.939 ECE=0.051  1145 ms/100 docs
bronze  acc=0.8990 macro_f1=0.8989 conf=0.780 ECE=0.121    11 ms/100 docs
```

`accuracy_report.py` also prints a confusion matrix, per-class precision/recall/F1,
and a 24-item curated probe set (21/24, 22/24 and 24/24 correct for
bronze/silver/gold). Its gate fails below 0.85 accuracy or macro-F1.

`evaluate.py` draws a random sample; `accuracy_report.py` samples per class. That is
why the two reports differ slightly: run-to-run variance is ~±0.5 points, and both
are shown rather than smoothing one into a rounder claim.

---

## Testing

```bash
python scripts/run_tests.py                 # 97 tests, project-local temp directory
ruff check src tests scripts               # lint
python scripts/smoke_test.py --base-url http://localhost:8000   # every endpoint
python scripts/ui_smoke.py --base-url http://localhost:8000     # real browser
```

`ui_smoke.py` drives Chromium against the running page: it files a story, renders
the explanation, records a correction, compares every ready desk, reads the
signals panel, and fails on any console error or HTTP response >= 400. It also
writes the README screenshots. Install once with `pip install playwright` and
`playwright install chromium`; CI runs it against the bronze-tier server.

The suite covers the API contract, tier routing and fallback, config validation,
preprocessing invariants, explainability, the feedback ledger, live insights,
the shipped UI assets, and degraded operation with no models installed.

---

## Project layout

```
├── src/newsclass/
│   ├── app.py             FastAPI routes, middleware, probe endpoints
│   ├── config.py          Typed settings from environment variables
│   ├── registry.py        Model discovery, routing, graceful fallback
│   ├── tiers.py           Bronze / Silver / Gold implementations
│   ├── train.py           Reproducible training for all three tiers
│   ├── text.py            Shared preprocessing (prevents train/serve skew)
│   ├── schemas.py         Pydantic request/response contracts
│   ├── explain.py         Per-tier attributions (linear exact, neural saliency)
│   ├── feedback.py        Append-only human-correction ledger
│   ├── stats.py           Bounded in-process traffic window for /v1/insights
│   ├── metrics.py         Prometheus instruments
│   ├── static/            Wire Desk UI + vendored Three.js/GSAP/Motion
│   └── logging_setup.py   Structured logging
├── tests/                 97 automated tests
├── scripts/               run_pipeline.py, accuracy_report.py, evaluate.py, smoke_test.py, ui_smoke.py
├── docker/                Dockerfile, compose stack, Prometheus config
├── docs/                  MODEL_CARD.md, risk policy, UI screenshots
├── artifacts/             Trained tiers + registry and audit JSON
└── *.ipynb                Two fully-executed research notebooks
```

---

## Notebooks (research record)

The notebooks document the experiments that produced these models. They are
deliverables in their own right and both run end-to-end with saved outputs.

- **`Word2Vec_LSTM_NLP_Classification.ipynb`** — AG News, own Word2Vec + BiLSTM,
  with an explicit TF-IDF baseline comparison.
- **`Transformer_DistilBERT_NLP_Classification.ipynb`** — BBC News, DistilBERT
  fine-tuning, including a measured discussion of `max_length` truncation.

---

## Honest limitations

Stating these up front is more credible than being caught out by them.

1. **The neural tiers do not dramatically beat the simple baseline on AG News.**
   TF-IDF + Logistic Regression reaches ~89.9%, the BiLSTM ~89.1%, and DistilBERT
   ~92.4%. On short, keyword-rich headlines the sequence model is not worth its
   latency; on longer copy the ranking can change, which is the point of a ladder.
2. **Run-to-run variance is ~±0.5 accuracy points.** Improvements smaller than that
   are noise. The project uses one fixed seed and one split, so confidence
   intervals are wider than the point estimates imply.
3. **Gold truncates at 64 tokens** (measured in `artifacts/gold/metrics.json`). For
   long articles most of the text is ignored.
   This is measured and stated in the notebook; raising it needs a GPU to stay
   practical.
4. **No authentication or rate limiting.** Put this behind a gateway before
   exposing it to untrusted traffic.
5. **English, four fixed classes, no "unknown".** Out-of-domain input is still
   forced into one of the four topics. The audit measures this: gold's mean
   confidence on out-of-domain probes is 0.77, versus 0.54 for bronze. A refusal
   policy would need a held-out OOD set and a threshold — it is not claimed here.

See `docs/MODEL_CARD.md` for the full risk and deployment policy.

---

## License

MIT. Datasets retain their original licenses (AG News and BBC News are used here
for academic purposes).
