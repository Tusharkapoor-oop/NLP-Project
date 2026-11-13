"""
BBC News Classifier API
=========================
FastAPI service for the Word2Vec + BiLSTM model.
"""
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.word2vec_lstm import TextClassifier


class PredictRequest(BaseModel):
    texts: List[str]
    top_k: int = 1


class PredictResponse(BaseModel):
    predictions: List[Dict]


# Lazy-loaded artifact globals
model = None
vocab = None
max_seq_len = None
label_classes = None


def load_artifacts(artifacts_dir: str = None):
    if artifacts_dir is None:
        artifacts_dir = ROOT / "models" / "word2vec_lstm"
    else:
        artifacts_dir = Path(artifacts_dir)
    meta_path = artifacts_dir / "deployment_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Deployment metadata not found at {meta_path}. Train the model first.")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    _vocab = meta["vocab"]
    _max_seq_len = meta["max_seq_len"]
    _embedding_dim = meta["embedding_dim"]
    _label_classes = meta["label_classes"]

    embedding_matrix = np.load(artifacts_dir / "embedding_matrix.npy").astype(np.float32)
    _model = TextClassifier(
        vocab_size=len(_vocab),
        embed_dim=_embedding_dim,
        hidden_dim=64,
        num_classes=len(_label_classes),
        embedding_matrix=embedding_matrix,
        num_layers=2,
        dropout=0.5,
    )
    state_path = artifacts_dir / "bilstm_bbc_classifier.pth"
    _model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    _model.eval()
    return _model, _vocab, _max_seq_len, _label_classes


app = FastAPI(
    title="BBC News Classifier API",
    version="1.0.0",
    description="Word2Vec + BiLSTM news classification",
)


@app.on_event("startup")
async def startup_event():
    global model, vocab, max_seq_len, label_classes
    model, vocab, max_seq_len, label_classes = load_artifacts()


def _preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text


def _text_to_sequence(text: str) -> List[int]:
    tokens = _preprocess(text).split()
    seq = [vocab.get(t, vocab["<OOV>"]) for t in tokens]
    if len(seq) >= max_seq_len:
        return seq[:max_seq_len]
    return seq + [vocab["<PAD>"]] * (max_seq_len - len(seq))


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head><title>BBC News Classifier API</title></head>
    <body style="font-family:Arial; max-width:600px; margin:40px auto; text-align:center;">
        <h1>📰 BBC News Classifier API</h1>
        <p>Word2Vec + BiLSTM model serving</p>
        <p><a href="/docs">Interactive API docs</a></p>
        <p><a href="/health">Health check</a></p>
    </body>
    </html>
    """)


@app.get("/health")
async def health():
    ready = model is not None
    return {
        "status": "ready" if ready else "loading",
        "model": "word2vec_bilstm_bbc",
        "classes": label_classes if ready else None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Train and restart the API.")
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    seqs = [_text_to_sequence(t) for t in req.texts]
    x = torch.LongTensor(seqs)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()

    results = []
    for prob in probs:
        top_idx = np.argsort(prob)[::-1][:req.top_k]
        results.append({
            "top_predictions": [
                {"label": label_classes[i], "confidence": round(float(prob[i]), 4)}
                for i in top_idx
            ]
        })
    return PredictResponse(predictions=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
