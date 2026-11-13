"""Silver layer: cleaned and standardized text.

Removes URLs, HTML, emails, numbers, punctuation; lowercases; tokenizes;
optionally removes stopwords.  Produces a pandas DataFrame that can be used
for classical ML / embedding training.
"""
import re
import json
import nltk
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd


class SilverPreprocessor:
    """Clean raw text into tokens + clean_text + label."""

    def __init__(self, stopwords_source: str = "nltk", stopwords_path: Optional[str] = None):
        self.stopwords_source = stopwords_source
        self.stopwords_path = stopwords_path
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        if self.stopwords_source == "none":
            return set()
        if self.stopwords_source == "nltk":
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        if self.stopwords_path:
            # space-separated stopwords in a file
            words = Path(self.stopwords_path).read_text(encoding="utf-8").split()
            return set(words)
        return set()

    def clean(self, text: str) -> Tuple[List[str], str]:
        if not isinstance(text, str):
            return [], ""
        text = text.lower()
        text = re.sub(r"http[s]?://\S+", " ", text)
        text = re.sub(r"www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = re.findall(r"\b[a-z]{2,}\b", text)
        tokens = [t for t in tokens if t not in self.stopwords]
        return tokens, " ".join(tokens)

    def transform_dataset(self, dataset, text_column: str, label_column: str, output_dir: str) -> pd.DataFrame:
        """Transform a HuggingFace DatasetDict to silver DataFrame."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for split in dataset.keys():
            for item in dataset[split]:
                text = item.get(text_column, "")
                label = item.get(label_column, "")
                if not isinstance(text, str) or not isinstance(label, str) or not text.strip():
                    continue
                tokens, clean_text = self.clean(text)
                records.append({
                    "split": split,
                    "text_raw": text.strip(),
                    "text_clean": clean_text,
                    "tokens": tokens,
                    "label": label.strip().lower(),
                })
        df = pd.DataFrame(records)
        df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
        df.to_csv(out_dir / "silver.csv", index=False)
        # per-split JSONL also for reproducibility
        for split in dataset.keys():
            split_df = df[df["split"] == split]
            split_df.to_json(out_dir / f"{split}_silver.jsonl", orient="records", lines=True, force_ascii=False)
        with open(out_dir / "silver_metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "num_rows": len(df),
                "vocab_size_approx": len(set(t for toks in df["tokens"] for t in toks)),
                "max_raw_len": int(df["text_raw"].str.split().str.len().max()),
                "max_clean_len": int(df["text_clean"].str.split().str.len().max()),
                "label_dist": df["label"].value_counts().to_dict(),
            }, f, indent=2, ensure_ascii=False)
        return df
