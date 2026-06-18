"""
Unit tests for the preprocessing pipeline.

All tests run on synthetic data — no real dataset downloads required.
"""
import hashlib
import pandas as pd
import pytest

from src.preprocessing.pipeline import (
    SCHEMA_COLS,
    check_no_leakage,
    clean_text,
    deduplicate,
    text_hash,
)


# ---------------------------------------------------------------------------
# clean_text edge cases
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_normal_string(self):
        result = clean_text("This is a normal sentence.")
        assert result == "This is a normal sentence."

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello world  ") == "hello world"

    def test_empty_string_returns_none(self):
        assert clean_text("") is None

    def test_too_short_returns_none(self):
        assert clean_text("abc") is None
        assert clean_text("a" * 9) is None

    def test_exactly_ten_chars_accepted(self):
        result = clean_text("a" * 10)
        assert result is not None

    def test_control_characters_removed(self):
        text = "hello\x01world\x1f!"
        result = clean_text(text)
        assert "\x01" not in result
        assert "\x1f" not in result
        assert "helloworld!" in result

    def test_newline_tab_preserved(self):
        # Newline (\n=0x0a) and tab (\t=0x09) are NOT control chars to strip
        result = clean_text("line one\nline two padded")
        assert result is not None
        assert "\n" in result

    def test_non_string_returns_none(self):
        assert clean_text(None) is None
        assert clean_text(42) is None

    def test_unicode_normalization(self):
        # NFC vs NFD: both should normalize to NFC
        import unicodedata
        nfd = unicodedata.normalize("NFD", "café au lait is nice!")
        result = clean_text(nfd)
        assert unicodedata.is_normalized("NFC", result)


# ---------------------------------------------------------------------------
# text_hash
# ---------------------------------------------------------------------------

class TestTextHash:
    def test_deterministic(self):
        assert text_hash("hello") == text_hash("hello")

    def test_different_strings(self):
        assert text_hash("hello") != text_hash("world")

    def test_returns_string(self):
        assert isinstance(text_hash("x" * 10), str)


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------

def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal schema-compliant DataFrame for testing."""
    n = len(next(iter(kwargs.values())))
    base = {col: ["x"] * n for col in SCHEMA_COLS}
    base.update(kwargs)
    return pd.DataFrame(base)


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        df = _make_df(text=["hello world!", "hello world!", "different text here"])
        result = deduplicate(df)
        assert len(result) == 2
        assert set(result["text"]) == {"hello world!", "different text here"}

    def test_no_duplicates_unchanged_length(self):
        df = _make_df(text=["alpha beta gamma", "delta epsilon zeta", "eta theta iota"])
        result = deduplicate(df)
        assert len(result) == 3

    def test_preserves_all_columns(self):
        df = _make_df(text=["unique text here one", "unique text here two"])
        result = deduplicate(df)
        assert list(result.columns) == SCHEMA_COLS

    def test_resets_index(self):
        df = _make_df(text=["dup text here now", "dup text here now", "other text abc"])
        result = deduplicate(df)
        assert list(result.index) == list(range(len(result)))

    def test_all_duplicates_leaves_one(self):
        df = _make_df(text=["same text repeated"] * 5)
        result = deduplicate(df)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# check_no_leakage
# ---------------------------------------------------------------------------

class TestLeakageCheck:
    def _df(self, texts: list[str]) -> pd.DataFrame:
        return _make_df(text=texts)

    def test_clean_passes(self):
        train = self._df(["unique train text a", "unique train text b"])
        test  = self._df(["unique test text c",  "unique test text d"])
        check_no_leakage(train, test, test_names=["test"])  # should not raise

    def test_overlap_raises(self):
        shared = "this text is in both train and test sets"
        train = self._df([shared, "another train-only text here"])
        test  = self._df([shared, "some test-only text here"])
        with pytest.raises(ValueError, match="Data leakage detected"):
            check_no_leakage(train, test, test_names=["test"])

    def test_multiple_test_sets_all_checked(self):
        train = self._df(["train only text abc"])
        test1 = self._df(["train only text abc"])  # overlap with train
        test2 = self._df(["test2 only text xyz"])
        with pytest.raises(ValueError, match="Data leakage detected"):
            check_no_leakage(train, test1, test2, test_names=["t1", "t2"])

    def test_empty_train_no_error(self):
        train = self._df([])
        test  = self._df(["some text here that's long"])
        check_no_leakage(train, test, test_names=["test"])  # should not raise
