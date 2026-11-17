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

- **One canonical split, shared by every tier.** `load_split` computes a single
  stratified partition from a fixed universe; `train_rows` subsamples *only* the
  training portion. A tier that trains on fewer rows is scored on exactly the
  same held-out documents as one that trains on more. This is enforced by
  `tests/test_data_integrity.py` (12 tests), because an earlier version of this
  project violated it: gold trained on one partition and was scored on another,
  and 23.2% of its "test set" was data it had already seen. That inflated its
  headline accuracy from 0.9035 to 0.9180.
- Fixed seed (42) and a fixed stratified split, so numbers are comparable across
  runs.
- Reported metrics: accuracy **and** macro-F1. Macro-F1 is authoritative because
  it weights every class equally. Cohen's κ and MCC are reported alongside,
  because both correct for chance agreement and are harder to flatter.
- **Every headline number carries a 95% bootstrap confidence interval**
  (`scripts/significance.py`, 2,000 resamples). A point estimate without an
  interval is not accepted as a claim.
- **Tier comparisons are paired.** All tiers see the same documents, so
  McNemar's test is the correct instrument, not a difference of independent
  proportions. `--require-order` makes "the docs claim a ladder" a hard gate.
- Run-to-run variation is **measured**, not assumed: `scripts/seed_variance.py`
  reports mean ± std across seeds. With `--vary-split` it measures full pipeline
  variance (bronze: std 0.0038, range 0.0075 over 3 seeds). Tiers that are
  deterministic under a fixed split are reported as *deterministic*, not as
  "zero variance" — the distinction matters, because a deterministic model says
  nothing about the variance of the pipeline around it.
- A trivial baseline (TF-IDF + Logistic Regression) is always reported alongside
  the neural models. On AG News this baseline is genuinely competitive, and
  saying so is part of the analysis.
- Calibration is measured (expected calibration error, 10 bins) and reported with
  a confusion matrix and per-class precision/recall/F1. Aggregate accuracy alone
  is not accepted as evidence of quality.
- **Training cost and serving cost are both reported.** Accuracy is not a
  sufficient criterion for deployment: gold costs 760× bronze per 1,000 requests
  and is 759× slower, for a difference the paired test cannot distinguish from
  chance (`scripts/benchmark.py`, `scripts/cost_model.py`).
- A 24-item hand-written probe set checks realistic headlines and flags
  out-of-domain copy as unscored. It is intentionally small and readable so every
  miss can be explained in review.

### Calibration under distribution shift (gold)

Gold is calibrated on AG News and overconfident on anything newer: on 149 live
2026 headlines it reported 0.90 while being right 74% of the time. A single
temperature cannot fix both regimes, so `src/newsclass/calibration.py` blends two
maps using a character n-gram gate that estimates "this text does not look like
AG News". AG-News-like text keeps its in-distribution confidence; modern news is
mapped by Platt scaling fitted on labelled 2026 headlines.

| Regime | ECE before | ECE after | Accuracy |
|---|---|---|---|
| 2026 headlines (149) | 0.152 | 0.067 (5-fold out-of-fold) | unchanged, 0.738 |
| AG News test (400) | 0.028 | 0.020 | unchanged, 0.920 |

### Adjudicated out-of-distribution accuracy (single reviewer)

Section labels are not topic labels. All 71 disputed documents from the 2026
headline set were reviewed one by one: the section label was upheld, corrected,
or the document was excluded as having no defensible single class (newsletter
promos, a property gallery, a cartoon). The ledger is
`artifacts/adjudication_ood.jsonl`; `scripts/adjudicate.py` applies it.

| Tier | As published | Adjudicated | 95% CI | Macro-F1 |
|---|---|---|---|---|
| gold | 0.7383 | **0.8214** (115/140) | 0.750-0.876 | 0.8066 |
| bronze | 0.7248 | 0.7786 (109/140) | 0.703-0.839 | 0.7393 |
| silver | 0.5839 | 0.6714 (94/140) | 0.590-0.744 | 0.6676 |

Where gold's 39 published errors went: 10 were pure label noise (the model's
answer was right and the section label wrong), 7 were excluded items, 3 errors
were revealed that the section label had hidden, leaving **25 genuine errors**.
Gold's Business recall rises from 0.44 to 0.64 and World precision to 1.00 after
correction; Sci/Tech precision stays low at 0.59, so the over-assignment of tech
is real, not a labelling artefact.

The corrected numbers still overlap: gold's interval [0.750, 0.876] contains
bronze's 0.779, so the ranking of gold over bronze on unseen news remains
unproven. This adjudication was performed by one reviewer (the assistant), not a
human panel; a second independent labeler is required before these figures are
treated as final.

Calibration never changes a predicted label, only how much it is trusted. The
artifact (`artifacts/gold/calibration.json`) is ignored by git because it lives
in the ignored `artifacts/gold/` directory: refit it with
`python scripts/fit_calibration.py` after training gold, and check `/v1/model`
for `calibration.gold.enabled`. If the artifact is absent or unreadable the tier
serves raw confidences, so a partial install still works.

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
