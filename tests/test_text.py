"""Tests for the preprocessing contract.

Training/serving skew is the failure mode these tests exist to prevent.
"""

from __future__ import annotations

import pytest

from newsclass.text import normalize_text, tokenize, transformer_input


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("BREAKING News") == "breaking news"

    def test_removes_urls(self):
        result = normalize_text("read more at https://example.com/story now")
        assert "https" not in result
        assert "example" not in result
        assert "read more at" in result

    def test_removes_html(self):
        assert "<p>" not in normalize_text("<p>hello</p>")

    def test_removes_digits(self):
        assert normalize_text("in 2024 the 3rd time") == "in the rd time"

    def test_collapses_whitespace(self):
        assert normalize_text("too    many\n\nspaces\t") == "too many spaces"

    def test_is_idempotent(self):
        once = normalize_text("Some TEXT with <b>tags</b> and 123 numbers!")
        assert normalize_text(once) == once

    def test_rejects_non_string(self):
        with pytest.raises(TypeError):
            normalize_text(None)  # type: ignore[arg-type]

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestTokenize:
    def test_drops_stopwords(self):
        tokens = tokenize("the market is rising")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "market" in tokens

    def test_drops_short_tokens(self):
        assert "a" not in tokenize("a test of the system")

    def test_lemmatizes(self):
        # "markets" -> "market"
        assert "market" in tokenize("markets are rising")

    def test_can_disable_stopwords(self):
        tokens = tokenize("the market", use_stopwords=False, use_lemma=False)
        assert "the" in tokens

    def test_empty_input(self):
        assert tokenize("") == []


class TestTransformerInput:
    def test_preserves_case(self):
        # Transformer tokenizers need natural casing; we must not lowercase here.
        assert transformer_input("Breaking NEWS") == "Breaking NEWS"

    def test_preserves_punctuation(self):
        assert "." in transformer_input("Hello world.")

    def test_strips_html(self):
        assert "<p>" not in transformer_input("<p>text</p>")

    def test_respects_max_chars(self):
        assert len(transformer_input("x" * 500, max_chars=100)) == 100
