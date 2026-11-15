"""Per-tier explainability: which words and tokens drove each prediction?

Every serious ML product eventually gets asked "why did it say that?". This
module gives an honest, per-tier answer using the method that is *native* to
each architecture — which is the only kind of explanation that is trustworthy:

* **bronze (TF-IDF + Logistic Regression)** — exact contributions. The model is
  linear, so ``coefficient[feature] * tfidf_value`` is the model's own
  arithmetic, not an approximation.
* **silver (Word2Vec + BiLSTM)** and **gold (DistilBERT)** — gradient×input
  saliency: propagate the class score back to the embedding/token-embedding
  inputs and attribute to each token the squared magnitude of its gradient.
  Higher = the token mattered more to this specific prediction.

The numbers are directional evidence, not proofs. We return the top-k tokens
with signed-ish weights where exact, plus a note stating the method used, so
nobody mistakes saliency for the model's real reasoning.
"""

from __future__ import annotations

from typing import Any

from .text import normalize_text, tokenize, transformer_input

MAX_EXPLAIN_CHARS = 4_000
TOP_K = 10


def _clean_for_length(text: str) -> str:
    return text[:MAX_EXPLAIN_CHARS]


# ---------------------------------------------------------------------------
# bronze: exact linear contributions
# ---------------------------------------------------------------------------
def explain_bronze(pipeline: Any, text: str, label_index: int, label_name: str) -> dict[str, Any]:
    """Exact per-word contributions of a linear TF-IDF classifier."""
    cleaned = normalize_text(_clean_for_length(text))

    vectorizer = pipeline.named_steps.get("tfidf") if hasattr(pipeline, "named_steps") else None
    classifier = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else None
    if vectorizer is None or classifier is None or not hasattr(classifier, "coef_"):
        return {"available": False, "reason": "artifact is not the expected linear pipeline"}

    row = vectorizer.transform([cleaned])
    _, columns = row.nonzero()
    feature_names = vectorizer.get_feature_names_out()

    # coef_ is [n_classes, n_features]; multi-class solvers expose one row per class.
    weights = classifier.coef_
    bias = float(classifier.intercept_[label_index]) if hasattr(classifier, "intercept_") else 0.0

    contributions = []
    for position in columns:
        feature = str(feature_names[position])
        value = float(row[0, position])
        coefficient = float(weights[label_index, position])
        contributions.append(
            {
                "token": feature,
                "contribution": round(coefficient * value, 5),
                "tfidf": round(value, 4),
                "coefficient": round(coefficient, 5),
            }
        )

    contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    top = contributions[:TOP_K]

    return {
        "available": True,
        "method": "exact linear contributions (coefficient × tf-idf)",
        "bias": round(bias, 5),
        "tokens": top,
        "context": (
            "The model is linear, so each word's contribution is exactly how much "
            "it pushed the score for the predicted class. Positive pushed towards "
            "the prediction, negative pushed against it."
        ),
    }


# ---------------------------------------------------------------------------
# silver: gradient × input saliency on the BiLSTM
# ---------------------------------------------------------------------------
def explain_silver(
    model: Any,
    torch_module: Any,
    encode_fn: Any,
    text: str,
    label_index: int,
) -> dict[str, Any]:
    """Saliency for the BiLSTM via gradient×input on the embedding outputs."""
    if model is None or torch_module is None or encode_fn is None:
        return {"available": False, "reason": "silver model is not loaded"}

    tokens = tokenize(_clean_for_length(text))[:64]
    ids = encode_fn(text)
    if not tokens:
        return {"available": False, "reason": "text produced no known tokens"}

    torch = torch_module
    tensor = torch.tensor([ids])
    model.eval()

    embedded = model.embedding(tensor)
    embedded = embedded.detach().requires_grad_(True)
    outputs, _ = model.lstm(embedded)
    mask = (tensor != 0).unsqueeze(-1).float()
    pooled = (outputs * mask).sum(1) / mask.sum(1).clamp(min=1)
    logits = model.classifier(model.dropout(pooled))
    score = logits[0, label_index]
    model.zero_grad()
    score.backward()

    grads = embedded.grad[0]  # [seq, dim]
    emb_values = embedded.detach()[0]
    # Gradient × input, L2 over dimensions: a standard, model-agnostic saliency.
    saliency = (grads * emb_values).pow(2).sum(-1).sqrt()

    total = float(saliency.sum()) or 1.0
    n_ids = len(ids)
    n_tokens = min(len(tokens), n_ids)

    scored = []
    for i in range(n_tokens):
        scored.append(
            {
                "token": tokens[i],
                "importance": round(float(saliency[i]) / total, 4),
            }
        )
    scored.sort(key=lambda item: item["importance"], reverse=True)

    return {
        "available": True,
        "method": "gradient × input saliency on the BiLSTM embedding layer",
        "tokens": scored[:TOP_K],
        "context": (
            "Each token's share of total saliency for this prediction. This is "
            "directional evidence from the model's gradients, not a proof of "
            "causation — treat it as a hint, not a verdict."
        ),
    }


# ---------------------------------------------------------------------------
# gold: token saliency for DistilBERT
# ---------------------------------------------------------------------------
def explain_gold(
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    text: str,
    label_index: int,
    max_length: int = 96,
) -> dict[str, Any]:
    """Saliency for the Transformer via gradient×input on the embedding output."""
    if model is None or tokenizer is None:
        return {"available": False, "reason": "gold model is not loaded"}

    cleaned = transformer_input(_clean_for_length(text))
    batch = tokenizer(cleaned, truncation=True, max_length=max_length, return_tensors="pt")
    model.eval()

    embedding_layer = model.get_input_embeddings()

    def _embed(inputs_ids):
        return embedding_layer(inputs_ids)

    input_ids = batch["input_ids"]
    embedded = _embed(input_ids)
    embedded = embedded.detach().requires_grad_(True)

    # Reconstruct the transformer input from the embedded sequence.
    kwargs = {k: v for k, v in batch.items() if k != "input_ids"}
    outputs = model(inputs_embeds=embedded, **kwargs)
    score = outputs.logits[0, label_index]
    model.zero_grad()
    score.backward()

    grads = embedded.grad[0]
    emb_values = embedded.detach()[0]
    saliency = (grads * emb_values).pow(2).sum(-1).sqrt()
    total = float(saliency.sum()) or 1.0

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    scored = []
    for i, tok in enumerate(tokens):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        scored.append({"token": tok, "importance": round(float(saliency[i]) / total, 4)})

    scored.sort(key=lambda item: item["importance"], reverse=True)
    return {
        "available": True,
        "method": "gradient × input saliency on the DistilBERT embedding output",
        "tokens": scored[:TOP_K],
        "context": (
            "Each sub-word's share of total saliency. WordPiece tokens may be "
            "split (e.g. '##ing'); read them as fragments of the words they came from."
        ),
    }
