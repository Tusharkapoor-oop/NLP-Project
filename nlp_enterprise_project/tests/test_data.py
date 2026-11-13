"""Unit tests for data processing layers."""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.silver import SilverPreprocessor
from features.gold import GoldBuilder


def test_preprocessor_basic():
    prep = SilverPreprocessor(stopwords_source="none")
    tokens, clean = prep.clean("Hello world! This is a TEST.")
    assert len(tokens) > 0
    assert clean


def test_tokenizer_sequence():
    prep = SilverPreprocessor(stopwords_source="none")
    _, clean1 = prep.clean("machine learning")
    _, clean2 = prep.clean("deep neural networks")
    series = __import__("pandas", fromlist=["Series"]).Series([["machine", "learning"], ["deep", "neural", "networks"]])
    gold = GoldBuilder(max_num_words=10, max_seq_len=5, embedding_dim=10)
    gold.fit_vocab(series)
    seq = gold.text_to_sequence(clean1 + " " + clean2)
    assert len(seq) == 5


if __name__ == "__main__":
    test_preprocessor_basic()
    test_tokenizer_sequence()
    print("All tests passed")
