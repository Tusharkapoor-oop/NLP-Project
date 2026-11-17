# Complete Project Pipeline Guide: From Zero to Production NLP

This document explains the entire News Topic Classification system from beginning to end. It is written so that anyone without prior knowledge of Machine Learning or Natural Language Processing (NLP) can understand what this system does, why it exists, and how every single piece works.

---

## Table of Contents

1. [The Real-World Problem: The Newsroom Flood](#1-the-real-world-problem-the-newsroom-flood)
2. [The Core Concept: Three Model Tiers (The Three Workers)](#2-the-core-concept-three-model-tiers-the-three-workers)
3. [The High-Level Architecture Overview](#3-the-high-level-architecture-overview)
4. [Phase 1: Data Ingestion and Train/Val/Test Splitting](#4-phase-1-data-ingestion-and-trainvaltest-splitting)
5. [Phase 2: Text Preprocessing (Cleaning the Raw Words)](#5-phase-2-text-preprocessing-cleaning-the-raw-words)
6. [Phase 3: Model Architecture and Training (Building the 3 Brains)](#6-phase-3-model-architecture-and-training-building-the-3-brains)
   - [Tier 1: Bronze (TF-IDF + Logistic Regression)](#tier-1-bronze-tf-idf--logistic-regression)
   - [Tier 2: Silver (Word2Vec + BiLSTM Neural Network)](#tier-2-silver-word2vec--bilstm-neural-network)
   - [Tier 3: Gold (DistilBERT Transformer)](#tier-3-gold-distilbert-transformer)
7. [Phase 4: Calibration and Out-of-Domain Guardrails](#7-phase-4-calibration-and-out-of-domain-guardrails)
8. [Phase 5: The Automated Release Gate (Quality Assurance)](#8-phase-5-the-automated-release-gate-quality-assurance)
9. [Phase 6: The Serving Layer (FastAPI Production Backend)](#9-phase-6-the-serving-layer-fastapi-production-backend)
10. [Phase 7: Explainability (Opening the Black Box)](#10-phase-7-explainability-opening-the-black-box)
11. [Phase 8: The Continuous Feedback Loop](#11-phase-8-the-continuous-feedback-loop)
12. [Phase 9: The Web User Interface (Wire Desk and Model Lab)](#12-phase-9-the-web-user-interface-wire-desk-and-model-lab)
13. [Phase 10: Production Verification, Testing, and Monitoring](#13-phase-10-production-verification-testing-and-monitoring)
14. [Summary Glossary of Key Terms](#14-summary-glossary-of-key-terms)

---

## 1. The Real-World Problem: The Newsroom Flood

### The Scenario
Every single minute, wire services (Reuters, Associated Press, Bloomberg) and digital publications publish thousands of articles across the globe.

A news agency or aggregator must route incoming stories to the right editorial desk immediately:
- A story about an election must go to **World News**.
- A story about a football final must go to **Sports**.
- A story about interest rates must go to **Business**.
- A story about a new computer processor must go to **Sci/Tech**.

### Why Humans Cannot Do This Alone
- **Volume**: A human takes 30 to 60 seconds to skim and tag an article. With 10,000 articles per hour, you would need an army of editors working 24/7.
- **Speed**: In financial and breaking news, being 5 seconds late means losing millions of readers or traders.
- **Consistency**: Different humans disagree on edge cases. A machine applies uniform logic every time.

### The Objective
Build an automated system that reads any English headline or article and outputs:
1. The correct category among the four classes (**World**, **Sports**, **Business**, **Sci/Tech**).
2. A confidence score between 0% and 100%.
3. An explanation showing which words triggered that decision.
4. A warning if the text is completely outside news topics (such as a cooking recipe).

---

## 2. The Core Concept: Three Model Tiers (The Three Workers)

A common mistake in AI projects is assuming you always use the largest, smartest model available. In real industry operations, large models are expensive, slow, and computationally heavy.

This project introduces a **Three-Tier Architecture**, comparable to three distinct workers in a newsroom:

```
+-------------------------------------------------------------------------+
|                              THE THREE DESKS                            |
+-------------------+--------------------+--------------------------------+
| Tier              | Model Technology   | Best Used For                  |
+-------------------+--------------------+--------------------------------+
| BRONZE            | TF-IDF +           | Bulk backfill, huge archives,  |
| (The Fast Intern) | Logistic Regression| 10,000 requests/sec, 0.83 MB   |
+-------------------+--------------------+--------------------------------+
| SILVER            | Word2Vec (CBOW) +  | Mid-volume feeds, edge servers,|
| (Junior Reporter) | BiLSTM Neural Net  | balance of speed and context   |
+-------------------+--------------------+--------------------------------+
| GOLD              | Fine-tuned         | User-facing decisions, breaking|
| (Senior Editor)   | DistilBERT         | high-stakes news, 91.5%+ acc   |
+-------------------+--------------------+--------------------------------+
```

### Analogy:
- **Bronze (The Intern with a Checklist)**: Scans for keywords. If it sees "goal" and "striker", it marks Sports. It cannot understand nuance, but it processes an article in 0.1 milliseconds.
- **Silver (The Junior Reporter)**: Reads sentences from start to finish and understands word relationships (e.g., "Paris" is to "France" what "Tokyo" is to "Japan"). Takes about 6 milliseconds.
- **Gold (The Senior Editor)**: Understands deep context, grammar, double meanings, and industry terminology. Knows that "Apple" in "Apple shares dropped" refers to the corporation, not the fruit. Takes about 70 milliseconds, but achieves maximum accuracy.

---

## 3. The High-Level Architecture Overview

The system is not just a script; it is a full software delivery pipeline:

```
[1. Dataset: AG News]
         |
         v
[2. Text Preprocessing: text.py]
         |
         +-------------------+-------------------+
         |                   |                   |
         v                   v                   v
   [Bronze Model]      [Silver Model]       [Gold Model]
    (TF-IDF + LR)      (W2V + BiLSTM)       (DistilBERT)
         |                   |                   |
         +-------------------+-------------------+
                             |
                             v
               [4. Shift Calibrator & Gate]
                             |
                             v
               [5. Release Gate Check (>85%)]
                             |
                             v
               [6. Model Registry (registry.json)]
                             |
                             v
               [7. FastAPI Backend Service]
                 /v1/predict   /v1/explain
                 /v1/feedback  /v1/insights
                             |
                             v
               [8. Interactive Web UI]
                 Wire Desk + Model Lab
```

---

## 4. Phase 1: Data Ingestion and Train/Val/Test Splitting

### What is the Dataset?
The system is trained on **AG News** (`SetFit/ag_news`), a standard international benchmark of over 120,000 news articles categorized into four balanced classes:
0: World
1: Sports
2: Business
3: Sci/Tech

### The Golden Rule of Machine Learning: Data Splitting
If a student sees the exam questions before taking the test, getting 100% does not prove intelligence; it proves memorization. In Machine Learning, this is called **data leakage**.

To prevent this, [src/newsclass/train.py](file:///c:/Users/ACER/Downloads/nlp%20project/src/newsclass/train.py#L100-L130) partitions the data into three separate piles using a fixed random seed (`seed=42`):

1. **Training Set (80% = 16,000 articles)**: The textbook. The models study this to learn patterns.
2. **Validation Set (10% = 2,000 articles)**: Practice quizzes. Used during training to check progress and select the best epoch without cheating.
3. **Test Set (10% = 2,000 articles)**: The final exam. Kept locked in a vault. Evaluated only after training is completely finished.

---

## 5. Phase 2: Text Preprocessing (Cleaning the Raw Words)

Computers do not understand letters; they only understand numbers. Before passing text to any model, it must be cleaned and standardized.

All cleaning rules are centralized in [src/newsclass/text.py](file:///c:/Users/ACER/Downloads/nlp%20project/src/newsclass/text.py) so that training data and live user inputs go through the exact same transformation.

### The Steps:
1. **HTML Entity Decoding**: Converts web clutter like `&amp;` to `&`, and `&quot;` to `"`.
2. **Whitespace Normalization**: Removes tabs, trailing spaces, and multiple newlines.
3. **Length Truncation**: Caps text at a maximum sequence length (e.g., 64 to 96 tokens) because news headlines convey 90% of their topic in the first two sentences.
4. **Tokenization**:
   - For Bronze: Basic word boundary splitting.
   - For Silver: Vocabulary dictionary matching words to integers (e.g., "apple" -> 421).
   - For Gold: Sub-word WordPiece tokenization (e.g., "unveiled" -> ["un", "##veil", "##ed"]).

---

## 6. Phase 3: Model Architecture and Training (Building the 3 Brains)

### Tier 1: Bronze (TF-IDF + Logistic Regression)
- **Concept**:
  - **TF (Term Frequency)**: How often does a word appear in this article?
  - **IDF (Inverse Document Frequency)**: How rare is this word across all news? Words like "the" and "is" appear everywhere, so their importance is reduced to near zero. Words like "quarterback" or "semiconductor" are rare and receive high weights.
- **Classifier**: Logistic Regression maps word scores to probabilities across the 4 classes.
- **Optimization**: L-BFGS solver running up to `max_iter=1000`.
- **Storage**: Compact `artifacts/bronze/model.joblib` file (only 0.83 MB).

### Tier 2: Silver (Word2Vec + BiLSTM Neural Network)
- **Concept**:
  - **Word2Vec (CBOW - Continuous Bag of Words)**: Words with similar meanings appear in similar contexts. It maps each word to a vector of 100 continuous numbers. "Football" and "Soccer" end up close together in mathematical space. Trained for **20 epochs**.
  - **BiLSTM (Bidirectional Long Short-Term Memory)**: A recurrent neural network that reads the sentence from left-to-right AND right-to-left.
- **Why Bidirectional?**
  Consider: *"Bank deposits were frozen after the collapse."* vs *"The river bank overflowed after the storm."*
  Reading only left-to-right, the word "bank" is ambiguous. Reading both directions reveals "deposits" or "river", disambiguating the meaning.
- **Training**: Trained for **6 epochs** in PyTorch. The model with the highest validation macro-F1 is automatically retained.

### Tier 3: Gold (DistilBERT Transformer)
- **Concept**:
  - A 66-million parameter Transformer model pre-trained on millions of books and Wikipedia pages.
  - Utilizes **Self-Attention**: every word attends to every other word simultaneously, capturing syntax, dependency, and deep semantic nuance.
- **Fine-Tuning**:
  - We freeze general English knowledge and train the top classification head specifically on AG News.
  - Trained for **2 epochs** using HuggingFace Trainer with AdamW optimizer, linear learning rate warmup, and weight decay.
- **Storage**: Weights saved in safe tensor format (`artifacts/gold/model.safetensors`, 268 MB).

---

## 7. Phase 4: Calibration and Out-of-Domain Guardrails

### What is Calibration?
If an AI tells you: *"I am 99% sure this article is about Sports"*, and it turns out to be wrong, it is **uncalibrated** (overconfident).
A well-calibrated model has an **Expected Calibration Error (ECE)** close to zero: whenever it predicts with 80% confidence, it should be correct 80% of the time.

- In our audit, the Gold model achieved an ECE of **0.026** (outstanding calibration).

### The Out-of-Domain (OOD) Problem
What happens if someone types a recipe: *"Whisk eggs and sugar until fluffy"*?
Because the model has only 4 buttons (World, Sports, Business, Sci/Tech), a naive model would be forced to pick one with high confidence.

To solve this, [scripts/fit_calibration.py](file:///c:/Users/ACER/Downloads/nlp%20project/scripts/fit_calibration.py) and [src/newsclass/calibration.py](file:///c:/Users/ACER/Downloads/nlp%20project/src/newsclass/calibration.py) implement a **Shift-Aware Calibrator**:
- It checks character n-grams and vocabulary distributions.
- When an article does not look like news, it applies a Platt-scaling sigmoid dampener.
- The confidence drops down to ~50%, signaling that human review is required.

---

## 8. Phase 5: The Automated Release Gate (Quality Assurance)

Before any model is allowed to serve real traffic, it must pass an automated inspection.

The script [scripts/evaluate.py](file:///c:/Users/ACER/Downloads/nlp%20project/scripts/evaluate.py) acts as a strict quality control inspector:
- It runs the newly trained model against the unseen held-out test split.
- **The Gate Threshold**: The model MUST achieve a **macro-F1 score of at least 0.85 (85%)**.
- If a model scores 0.849, the pipeline halts immediately, returns exit code 1, and refuses to boot the server.
- This prevents degraded or broken models from ever reaching users.

---

## 9. Phase 6: The Serving Layer (FastAPI Production Backend)

Once verified, models are loaded into a production REST API built with **FastAPI** ([src/newsclass/app.py](file:///c:/Users/ACER/Downloads/nlp%20project/src/newsclass/app.py)):

### Key API Endpoints:
- `POST /v1/predict`: Accepts `{"text": "headline"}` and returns class, probabilities, latency, and model version.
- `POST /v1/explain`: Returns the token-level attribution breakdown for that classification.
- `POST /v1/feedback`: Allows an editor to submit a correction if they believe a prediction was wrong.
- `GET /v1/insights`: Returns rolling 10-minute traffic metrics (p50/p95 latency, class distribution, drift).
- `GET /healthz`: Kubernetes liveness probe (checks if process is alive).
- `GET /readyz`: Kubernetes readiness probe (checks if model weights are loaded and ready to infer).

### Fault Tolerance & Graceful Fallback:
If a request comes in asking for the Gold tier, but the Gold model file is missing or busy, the system **automatically falls back to Silver or Bronze**. The end user never receives a 500 internal server error.

---

## 10. Phase 7: Explainability (Opening the Black Box)

In high-stakes industries, saying *"The AI said so"* is unacceptable. Editors and auditors demand to know **why**.

The system includes built-in explainability ([src/newsclass/explain.py](file:///c:/Users/ACER/Downloads/nlp%20project/src/newsclass/explain.py)):

1. **For Bronze (Exact Linear Contributions)**:
   - Mathematics: $\text{Weight} = \text{TF-IDF Value} \times \text{Logistic Regression Coefficient}$.
   - Every single word's mathematical contribution to the final score is calculated exactly.
2. **For Silver and Gold (Gradient x Input Saliency)**:
   - Neural networks do not have single linear coefficients.
   - The engine computes the gradient of the predicted class score with respect to the input word embeddings.
   - Words that pushed the classification the hardest light up in green; words pushing toward other classes light up in red.

---

## 11. Phase 8: The Continuous Feedback Loop

No machine learning model is perfect. When an editor spots a misclassified headline, they click **"Send correction"** in the UI.

1. The API validates the submission via [src/newsclass/feedback.py](file:///c:/Users/ACER/Downloads/nlp%20project/src/newsclass/feedback.py).
2. The corrected label and headline are recorded in an append-only JSON Lines file (`artifacts/feedback.jsonl`).
3. During the next retraining cycle, this dataset of human feedback is ingested to eliminate previous blind spots.

---

## 12. Phase 9: The Web User Interface (Wire Desk and Model Lab)

The project ships with a complete, fully offline, responsive frontend interface:

1. **The Wire Desk (`/`)**:
   - A dark-mode newsroom interface inspired by financial terminals.
   - Built with local vendored libraries ([Three.js](file:///c:/Users/ACER/Downloads/nlp%20project/src/newsclass/static/vendor/three.min.js), [GSAP](file:///c:/Users/ACER/Downloads/nlp%20project/src/newsclass/static/vendor/gsap.min.js), [Motion](file:///c:/Users/ACER/Downloads/nlp%20project/src/newsclass/static/vendor/motion.min.js)).
   - Features zero external CDN calls; operates completely offline in air-gapped environments.
   - Includes real-time classification, probability distribution bars, and live explanation badges.

2. **The Model Lab (`/lab`)**:
   - A multi-tier comparative bench.
   - Submits the exact same article to **Bronze, Silver, and Gold simultaneously**.
   - Displays prediction labels, confidence distributions, and server latencies side-by-side to visualize speed vs. accuracy trade-offs.

---

## 13. Phase 10: Production Verification, Testing, and Monitoring

Software quality is guaranteed through three levels of testing:

1. **Unit and Contract Tests (`scripts/run_tests.py`)**:
   - 103 automated tests run across `tests/`.
   - Checks API status codes, schema validation, tier fallbacks, UI asset integrity, and calibration math.
2. **Containerized Smoke Tests (`scripts/smoke_test.py`)**:
   - Automatically probes all endpoints on live servers to verify end-to-end integration.
3. **Observability and Metrics**:
   - Prometheus metrics endpoint (`/metrics`) tracks `newsclass_predictions_total`, latency histograms, and fallback counters.
   - Grafana dashboards visualize operations in real time.

---

## 14. Summary Glossary of Key Terms

| Term | Simple Definition |
|---|---|
| **NLP** | Natural Language Processing — teaching computers to read and understand human language. |
| **TF-IDF** | A formula that measures how important a word is to a document compared to a whole library of text. |
| **Word2Vec** | Turning words into lists of numbers so that words with similar meanings are close to each other. |
| **LSTM** | Long Short-Term Memory — a type of neural network that remembers earlier words in a sentence. |
| **Transformer** | A modern neural network architecture that processes all words simultaneously using attention mechanisms. |
| **DistilBERT** | A smaller, faster version of Google's famous BERT language model. |
| **Epoch** | One complete pass of the machine learning algorithm through the entire training dataset. |
| **Macro-F1** | An accuracy score that treats every class equally, preventing the model from cheating on rare classes. |
| **Calibration** | Ensuring that when a model claims 90% confidence, it is genuinely right 90% of the time. |
| **Inference** | The moment the trained model receives new text and generates a prediction. |
| **Fallback** | Automatically routing a request to a backup model if the primary model fails or is unavailable. |
