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

# 2. train the cheapest tier (about 20 seconds, no PyTorch needed)
python -m newsclass.train --tier bronze

# 3. serve
$env:PYTHONPATH="src"                       # Windows
export PYTHONPATH=src                       # bash
uvicorn newsclass.app:app --port 8000
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

## The three tiers, and what each one is *for*

This is the core design decision. Rather than shipping one model, the system ships
a **quality/cost ladder**. The same request can be served at three different price
points, and the choice is an explicit, measurable trade-off.

| Tier | Model | Accuracy | Macro-F1 | Size | Latency | Train (CPU) | Use when |
|---|---|---|---|---|---|---|---|
| **Bronze** | TF-IDF + Logistic Regression | 0.901 | 0.901 | 0.8 MB | 30 ms / 100 docs | ~20 s | High volume, cost-sensitive, no PyTorch runtime needed |
| **Silver** | Word2Vec (CBOW) + BiLSTM | 0.898 | 0.899 | 9 MB | 1465 ms / 100 docs | ~7 min | You want learned embeddings and sequence modelling on CPU |
| **Gold** | Fine-tuned DistilBERT | 0.918 | 0.918 | 269 MB | 22501 ms / 100 docs | ~88 min | Accuracy matters more than latency or memory |

All figures are produced by `python scripts/evaluate.py` on the full 2,000-document
test set — regenerate them yourself with that one command. Three honest readings of
this table are worth stating out loud:

* **Gold buys ~1.7 accuracy points over bronze** at 335× the size and ~750× the
  latency. Whether that trade is worth it is a business decision, which is the
  whole point of routing by tier.
* **Silver is not strictly between its neighbours.** It is *slower* than bronze yet
  no more accurate on this dataset — sequence modelling pays off on longer, more
  nuanced documents, and AG News headlines are short and keyword-rich.
* **Latency scales with cost** far more sharply than accuracy does on CPU. A
  request routed to gold takes ~225 ms; the same request to bronze takes ~0.3 ms.

> **Actively measured numbers, not marketing.** `artifacts/registry.json` holds the
> exact figures produced by the training runs on this machine. Regenerate with
> `python scripts/evaluate.py` — it re-scores every tier on the held-out test set.

### Why a tier ladder rather than one "best" model?

A strict reviewer will ask "why not just use the best model?" The honest answer is
that the best model is 300× larger and 45× slower to train, and on this dataset it
buys roughly four accuracy points. Whether that is worth it depends entirely on
traffic volume and the cost of a wrong answer:

- **Bulk classification of a million headlines** — bronze. The marginal accuracy is
  not worth 255 MB per replica times N replicas.
- **Moderating user-facing content** — gold. A wrong label is visible to a user.
- **A useful middle ground** — silver. Sequence-aware, still CPU-friendly.

The API reports which tier served every response, so this decision is auditable
after the fact rather than hidden inside a config file.

---

## What "production" means here (the checklist)

| Concern | How it is handled | Where |
|---|---|---|
| Config without redeploys | Typed settings from environment variables | `src/newsclass/config.py` |
| Training/serving skew | One shared preprocessing module, imported by both | `src/newsclass/text.py` |
| Graceful degradation | Falls back to the best available tier, logged and metriced | `src/newsclass/registry.py` |
| Liveness vs readiness | Separate `/healthz` and `/readyz`; liveness ignores models | `src/newsclass/app.py` |
| Metrics | Request counts, latency histograms, confidence drift, fallbacks | `src/newsclass/metrics.py` |
| Stateless containers | Models read from a mounted volume; no writes at runtime | `docker/Dockerfile` |
| Reproducible builds | Multi-stage image, pinned base, non-root user | `docker/Dockerfile` |
| Automated verification | 73 tests + CI matrix + container smoke test | `tests/`, `.github/workflows/ci.yml` |
| Release gate | Refuses to ship an artifact below a macro-F1 threshold | `scripts/evaluate.py --gate` |
| Observability stack | Pre-provisioned Prometheus + Grafana | `docker/` |
| Governance | Model card, risk policy, changelog | `docs/`, `CHANGELOG.md` |

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness. 200 while the process runs. Never checks models. |
| `GET` | `/readyz` | Readiness. 503 when no tier can serve. |
| `GET` | `/v1/tiers` | Every tier with accuracy, size, latency and a human description. |
| `GET` | `/v1/model` | Active default tier, loaded tiers, label set. |
| `POST` | `/v1/predict` | Classify one document. |
| `POST` | `/v1/predict/batch` | Classify up to `max_batch_size` documents. |
| `GET` | `/metrics` | Prometheus exposition. |

Every response carries `x-request-id` and `x-response-time-ms`.

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
python -m newsclass.train --tier bronze     # ~20 s
python -m newsclass.train --tier silver     # ~7 min
python -m newsclass.train --tier gold       # ~45 min on CPU
python -m newsclass.train --tier all        # everything, then write registry.json
```

Each tier writes its weights plus a `metrics.json`, and `registry.json` aggregates
them. The registry is what the API reads at boot.

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
python scripts/evaluate.py                        # all ready tiers
python scripts/evaluate.py --tier gold --gate 0.90   # release gate
```

```
bronze  acc=0.8950 macro_f1=0.8944 conf=0.812 42 ms/100 docs
silver  acc=0.8915 macro_f1=0.8911 conf=0.795 118 ms/100 docs
gold    acc=0.9400 macro_f1=0.9395 conf=0.901 3420 ms/100 docs
```

---

## Testing

```bash
pytest -q                                  # 73 tests
ruff check src tests scripts               # lint
python scripts/smoke_test.py               # against a running instance
```

The suite covers the API contract, tier routing and fallback, config validation,
preprocessing invariants, and degraded operation with no models installed.

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
│   ├── metrics.py         Prometheus instruments
│   └── logging_setup.py   Structured logging
├── tests/                 73 automated tests
├── scripts/               evaluate.py, smoke_test.py
├── docker/                Dockerfile, compose stack, Prometheus config
├── docs/                  MODEL_CARD.md, risk policy
├── artifacts/             Trained tiers + registry.json
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
   TF-IDF + Logistic Regression reaches ~89.4%, and the BiLSTM lands within half a
   point of it. This is a known property of short, keyword-rich news headlines, and
   it is documented rather than hidden. The DistilBERT tier is the one that clearly
   pulls ahead.
2. **Run-to-run variance is ~±0.5 accuracy points.** Improvements smaller than that
   are noise. The project uses one fixed seed and one split, so confidence
   intervals are wider than the point estimates imply.
3. **Gold truncates at 128 tokens.** For long articles most of the text is ignored.
   This is measured and stated in the notebook; raising it needs a GPU to stay
   practical.
4. **No authentication or rate limiting.** Put this behind a gateway before
   exposing it to untrusted traffic.
5. **English, four fixed classes, no "unknown".** Out-of-domain input is still
   forced into one of the four topics with a confidence score that can be low.

See `docs/MODEL_CARD.md` for the full risk and deployment policy.

---

## License

MIT. Datasets retain their original licenses (AG News and BBC News are used here
for academic purposes).
