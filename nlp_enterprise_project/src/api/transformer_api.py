"""
AG News DistilBERT API
=======================
FastAPI service for the DistilBERT fine-tuned model.
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class PredictRequest(BaseModel):
    texts: List[str]
    top_k: int = 1


class PredictResponse(BaseModel):
    predictions: List[Dict]


model = None
tokenizer = None


def load_artifacts(artifacts_dir: str = None):
    if artifacts_dir is None:
        artifacts_dir = ROOT / "models" / "transformer" / "best"
    else:
        artifacts_dir = Path(artifacts_dir)
    if not (artifacts_dir / "config.json").exists():
        raise FileNotFoundError(f"Transformer artifacts not found at {artifacts_dir}. Train the model first.")
    _tokenizer = AutoTokenizer.from_pretrained(str(artifacts_dir), local_files_only=True)
    _model = AutoModelForSequenceClassification.from_pretrained(str(artifacts_dir), local_files_only=True)
    _model.eval()
    return _model, _tokenizer


app = FastAPI(
    title="AG News DistilBERT API",
    version="1.0.0",
    description="DistilBERT fine-tuned on AG News",
)


@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    model, tokenizer = load_artifacts()


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head><title>AG News DistilBERT API</title></head>
    <body style="font-family:Arial; max-width:600px; margin:40px auto; text-align:center;">
        <h1>🗞️ AG News DistilBERT API</h1>
        <p>Transformer-based news classification</p>
        <p><a href="/docs">Interactive API docs</a></p>
        <p><a href="/health">Health check</a></p>
    </body>
    </html>
    """)


@app.get("/health")
async def health():
    ready = model is not None
    id2label = model.config.id2label if ready else None
    return {
        "status": "ready" if ready else "loading",
        "model": "distilbert-base-uncased",
        "classes": id2label,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Train and restart the API.")
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    inputs = tokenizer(req.texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()

    id2label = model.config.id2label
    results = []
    for prob in probs:
        top_idx = np.argsort(prob)[::-1][:req.top_k]
        results.append({
            "top_predictions": [
                {"label": id2label[str(i)], "confidence": round(float(prob[i]), 4)}
                for i in top_idx
            ]
        })
    return PredictResponse(predictions=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
