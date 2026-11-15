# Model Risk & Deployment Policy

A short, explicit policy so that deploying a model is a checklist and not a
judgement call. Reviewers should be able to point at each section.

## 1. Tier definitions

| Tier | Model | Intended traffic | Rationale |
|---|---|---|---|
| **bronze** | TF-IDF + Logistic Regression | High volume, low stakes, bulk backfill | Sub-millisecond inference, 0.83 MB, no PyTorch runtime. |
| **silver** | Word2Vec (CBOW) + BiLSTM | Medium volume, general classification | Learns domain embeddings; strong accuracy at CPU cost. |
| **gold** | fine-tuned DistilBERT | Low volume, high stakes, user-facing | Highest accuracy; needs the heaviest runtime. |

Routing is by `NEWSCLASS_DEFAULT_TIER`. A caller may request a tier only when
`NEWSCLASS_ALLOW_TIER_OVERRIDE=true`.

## 2. Promotion gates

A tier may not serve production traffic until **all** of the following hold:

1. `python scripts/run_tests.py` passes in full (101+ tests).
2. `python scripts/evaluate.py --tier <t> --gate 0.85` passes on the held-out
   test set.
3. `python scripts/accuracy_report.py --tier <t>` passes: accuracy and macro-F1
   above 0.85, per-class scores inspected, and the curated probe set above 70%.
4. `artifacts/registry.json` contains an entry for the tier with `accuracy`
   and `macro_f1` populated.
5. `scripts/smoke_test.py` passes against a running container.
6. The change is recorded in `CHANGELOG.md`.

The 0.85 macro-F1 gate is a floor, not a target. It exists to catch a broken
artifact, not to certify quality.

## 3. Data governance

- Training data is **AG News** (`SetFit/ag_news`), a public academic benchmark
  containing English news headlines. It is not customer data.
- No personal data is written to logs. The API logs request metadata (path,
  status, latency, tier) and never the request body.
- Preprocessing is centralised in `src/newsclass/text.py`. Training and serving
  import the same functions; divergence is treated as a defect.
- Vocabulary and Word2Vec vectors for the silver tier are built **only** from the
  training split, so the test set cannot leak into the model.

## 4. Evaluation protocol

- Fixed seed (42) and a fixed stratified split, so numbers are comparable across
  runs.
- Reported metrics: accuracy **and** macro-F1. Macro-F1 is authoritative because
  it weights every class equally.
- Run-to-run variation is approximately ±0.5 accuracy points. Any claimed
  improvement smaller than that is not evidence and must not be presented as one.
- A trivial baseline (TF-IDF + Logistic Regression) is always reported alongside
  the neural models. On AG News this baseline is genuinely competitive, and
  saying so is part of the analysis.
- Calibration is measured (expected calibration error, 10 bins) and reported with
  a confusion matrix and per-class precision/recall/F1. Aggregate accuracy alone
  is not accepted as evidence of quality.
- A 24-item hand-written probe set checks realistic headlines and flags
  out-of-domain copy as unscored. It is intentionally small and readable so every
  miss can be explained in review.

## 5. Monitoring in production

| Signal | Metric | Action |
|---|---|---|
| Traffic | `newsclass_predictions_total` | Alert on a sustained drop (traffic loss). |
| Latency | `newsclass_prediction_latency_seconds` | Alert if p95 exceeds the SLO. |
| Drift | `newsclass_prediction_confidence` | Investigate if the distribution shifts markedly. |
| Degradation | `newsclass_tier_fallbacks_total` | Treat any sustained increase as an incident. |

A sudden fall in mean confidence usually means the input distribution changed.
It does **not** by itself mean the model is wrong, so it is a prompt to look, not
an automatic rollback.

## 6. Rollback

Model artifacts are versioned in `registry.json`. To roll back:

1. Restore the previous `artifacts/` directory (or point
   `NEWSCLASS_ARTIFACTS_DIR` at the previous copy).
2. Restart the service. Models load at startup.
3. Confirm `/readyz` returns 200 and `newsclass_tier_fallbacks_total` is flat.

Because the image reads artifacts from a mounted volume, rollback does not
require a rebuild.

## 7. Known limitations (must be disclosed)

- **Truncation.** The gold tier reads only the first 64 sub-word tokens (see
  `artifacts/gold/metrics.json`). On long articles most of the document is
  ignored. Measured and stated in the notebook.
- **Single split.** Metrics come from one train/val/test split. No cross-validation
  or multi-seed averaging, so confidence intervals are wider than the point
  estimates suggest.
- **No authentication or rate limiting.** This service must sit behind an API
  gateway before it is exposed to untrusted traffic.
- **English only.** The preprocessing and the pre-trained model are English.
- **Topic taxonomy is fixed** to the four AG News classes. Out-of-scope input is
  still forced into one of them with a confidence score; there is no "unknown"
  class. The audit measures the gap: mean confidence on out-of-domain probes is
  0.77 for gold and 0.54 for bronze. A production refusal policy needs a dedicated
  OOD set and a threshold; that is out of scope for this release.
- **Explanations are evidence, not proof.** Linear contributions are exact; neural
  saliency is directional. The API names the method in every response, and a
  failed explanation degrades to `unavailable` rather than failing the prediction.
- **Feedback is unmoderated training data.** The append-only ledger stores the
  submitted text and verdict; it is not automatically applied to the model, and it
  must be reviewed before the next retraining run. Validation errors never echo
  the submitted document back to the caller.
