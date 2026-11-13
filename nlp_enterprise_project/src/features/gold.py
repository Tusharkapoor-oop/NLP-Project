"""Gold layer: model-ready features.

Turns cleaned text into sequences / embeddings, encoded labels, stratified splits,
and an embedding matrix for neural models.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gensim


# Constants
PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"


class GoldBuilder:
    """Build gold-level features: token sequences, embedding matrix, labels."""

    def __init__(self, max_num_words: int, max_seq_len: int, embedding_dim: int):
        self.max_num_words = max_num_words
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.word_to_idx: Dict[str, int] = {}
        self.label_encoder = LabelEncoder()

    def fit_vocab(self, tokens_series: pd.Series):
        word_freq = Counter([t for toks in tokens_series for t in toks])
        vocab = {PAD_TOKEN: 0, OOV_TOKEN: 1}
        for word, _ in word_freq.most_common(self.max_num_words):
            if word not in vocab:
                vocab[word] = len(vocab)
        self.word_to_idx = vocab

    def text_to_sequence(self, text: str) -> List[int]:
        tokens = text.split()
        seq = [self.word_to_idx.get(t, self.word_to_idx[OOV_TOKEN]) for t in tokens]
        if len(seq) >= self.max_seq_len:
            return seq[:self.max_seq_len]
        return seq + [self.word_to_idx[PAD_TOKEN]] * (self.max_seq_len - len(seq))

    def build_embedding_matrix(self, word2vec_path: Optional[str]) -> np.ndarray:
        if word2vec_path and Path(word2vec_path).exists():
            w2v = gensim.models.Word2Vec.load(word2vec_path)
        else:
            raise FileNotFoundError(f"Word2Vec model not found at {word2vec_path}")
        vocab_size = len(self.word_to_idx)
        matrix = np.random.normal(scale=0.05, size=(vocab_size, self.embedding_dim)).astype(np.float32)
        matrix[0] = np.zeros(self.embedding_dim)
        matrix[1] = np.zeros(self.embedding_dim)
        for word, idx in self.word_to_idx.items():
            if word in w2v.wv:
                matrix[idx] = w2v.wv[word]
        return matrix

    def transform(self, df: pd.DataFrame, label_column: str = "label", text_column: str = "text_clean",
                  stratify_col: str = "label", seed: int = 42) -> Tuple[np.ndarray, ...]:
        self.fit_vocab(df["tokens"])
        X = np.array([self.text_to_sequence(t) for t in df[text_column]])
        y = self.label_encoder.fit_transform(df[label_column])
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save(self, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.word_to_idx, f, indent=2, ensure_ascii=False)
        with open(out / "label_classes.json", "w", encoding="utf-8") as f:
            json.dump(self.label_encoder.classes_.tolist(), f, indent=2, ensure_ascii=False)
