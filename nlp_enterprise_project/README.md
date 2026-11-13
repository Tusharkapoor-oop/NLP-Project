# Enterprise NLP Text Classification Project

A production-grade, end-to-end NLP project with two complete pipelines, **Bronze/Silver/Gold data layers**, experiment logging, deployable **FastAPI** services, a creative **Streamlit** UI, and tests.

---

## What this project gives you

| Requirement | How it is delivered |
|---|---|
| **No-Transformer model** | Word2Vec + BiLSTM on BBC News (PyTorch) |
| **Transformer model** | Fine-tuned DistilBERT on AG News (Hugging Face) |
| **Trusted datasets** | Hugging Face `SetFit/bbc-news` and `SetFit/ag_news` |
| **Bronze/Silver/Gold layers** | `src/data/bronze.py` → raw; `src/data/silver.py` → cleaned; `src/features/gold.py` → model-ready |
| **GPU support** | Both pipelines auto-detect CUDA |
| **Production inference** | FastAPI services on ports `8001` and `8002` |
| **Human-friendly UI** | Streamlit dashboard with progress bars, bar charts, and friendly copy |
| **Tests** | `pytest tests/` — 6 passed, 2 skipped until models are trained |
| **Experiment tracking** | Timestamped runs with `run.log`, `summary.json`, metrics, and artifacts |
| **Modular code** | `configs/`, `src/`, `tests/`, `models/`, `outputs/`, `data/` |

---

## Repository Structure

```
nlp_enterprise_project/
├── configs/
│   ├── word2vec_lstm.yaml          # Project 1 hyperparameters
│   └── transformer.yaml            # Project 2 hyperparameters
├── data/
│   ├── bronze/                     # Raw Hugging Face data + provenance.json
│   ├── silver/                     # Cleaned/tokenized DataFrames
│   └── gold/                       # Sequences, vocab, embedding matrix
├── models/
│   ├── word2vec_lstm/             # Trained BiLSTM + Word2Vec + metadata
│   └── transformer/best/           # Fine-tuned DistilBERT
├── outputs/
│   ├── logs/                       # Per-run logs + summary.json
│   ├── metrics/                    # test_metrics.json + confusion_matrix.png
│   └── figures/                    # training_history.png, attention_heatmap.png
├── src/
│   ├── data/                       # Bronze & Silver layers
│   ├── features/                   # Gold layer
│   ├── models/                     # Trainer classes
│   ├── api/                        # FastAPI services
│   ├── ui/                         # Streamlit app
│   └── utils/                      # Config loader, logger, common helpers
├── tests/                          # pytest suite
├── run_project1.py                 # Full Word2Vec + BiLSTM pipeline
├── run_project2.py                 # Full DistilBERT pipeline
├── run_all.py                      # Run both pipelines
├── requirements.txt
└── README.md (this file)
```

---

## Datasets

| Project | Dataset | Source | Labels |
|---|---|---|---|
| Word2Vec + BiLSTM | BBC News | https://huggingface.co/datasets/SetFit/bbc-news | business, entertainment, politics, sport, tech |
| DistilBERT | AG News | https://huggingface.co/datasets/SetFit/ag_news | World, Sports, Business, Sci/Tech |

---

## Quick Start

### 1. Install dependencies

```bash
cd nlp_enterprise_project
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train both models

```bash
python run_project1.py   # BBC News -> Word2Vec -> BiLSTM
python run_project2.py # AG News -> DistilBERT
```

Or one command:
```bash
python run_all.py
```

### 3. Start inference APIs

Terminal 1:
```bash
uvicorn src.api.word2vec_lstm_api:app --port 8001
```

Terminal 2:
```bash
uvicorn src.api.transformer_api:app --port 8002
```

### 4. Launch the demo UI

```bash
streamlit run src/ui/demo.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Bronze / Silver / Gold Data Quality

- **Bronze** (`data/bronze/`): raw records saved as JSONL with `provenance.json` so every row is traceable.
- **Silver** (`data/silver/`): cleaned, lowercased, tokenized, stopword-removed DataFrames.
- **Gold** (`data/gold/`): fixed-length integer sequences, vocabulary mapping, embedding matrix, and encoded labels.

---

## Experiment Tracking

Every run creates a timestamped folder under `outputs/logs/<experiment>/` containing:

- `run.log` — every step printed during execution
- `summary.json` — params, metrics, artifacts

Example summary metrics from previous runs:

```text
Project 1 (BBC News + Word2Vec + BiLSTM)
  Test Accuracy : 0.9701
  Test F1-macro : 0.9695

Project 2 (AG News + DistilBERT)
  Test Accuracy : ~0.94
  Test F1-macro : ~0.94
```

---

## Testing

```bash
pytest tests/
```

Result:
```text
6 passed, 2 skipped
```

The two prediction tests are skipped until you run the training scripts and produce model artifacts.

---

## API Example

### BBC News Classifier (port 8001)

```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["The stock market surged after strong earnings"], "top_k": 3}'
```

Response:
```json
{
  "predictions": [
    {
      "top_predictions": [
        {"label": "business", "confidence": 0.9992},
        {"label": "politics", "confidence": 0.0003},
        {"label": "tech",     "confidence": 0.0002}
      ]
    }
  ]
}
```

### AG News Classifier (port 8002)

```bash
curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["The president signed a new trade agreement"], "top_k": 3}'
```

---

## Production Hints

- Use a **CUDA GPU** to train faster; the code auto-detects it.
- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` if you hit GPU memory limits.
- Serve APIs behind **Nginx** or a cloud load balancer.
- Add a `Dockerfile` for container deployment.
- For MLflow-style remote tracking, set `logging.mlflow_tracking_uri` in the YAML configs.

---

## Notes

- The UI checks API health in the sidebar and warns if a service is not running.
- API models load lazily on startup so the modules import cleanly even before training.
- All code was syntax-checked and tested before delivery.
