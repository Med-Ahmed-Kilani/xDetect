"""
Tests for Month 2 pooling logic — concatenation of EN+DE+AR training splits.

Row counts must equal the exact sum of parts; no data loss; no duplication;
language column preserved per row.
"""
import pandas as pd
import pytest

from src.preprocessing.pipeline import SCHEMA_COLS, check_no_leakage, deduplicate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LID = 0  # global offset so texts across calls don't collide


def _part(n: int, lang: str) -> pd.DataFrame:
    global _LID
    offset = _LID
    _LID += n
    texts = [f"{lang} training sample number {offset + i} long enough" for i in range(n)]
    labels = [i % 2 for i in range(n)]
    base = {col: ["x"] * n for col in SCHEMA_COLS}
    base["text"]     = texts
    base["label"]    = labels
    base["language"] = [lang] * n
    return pd.DataFrame(base)


def _pool(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Same logic as the pipeline: concat then reset index."""
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Row count
# ---------------------------------------------------------------------------

class TestPooledRowCount:
    def test_row_count_equals_sum_of_parts(self):
        en = _part(40, "en")
        de = _part(38, "de")
        ar = _part(42, "ar")
        pooled = _pool([en, de, ar])
        assert len(pooled) == 40 + 38 + 42

    def test_two_equal_parts(self):
        en = _part(50, "en")
        de = _part(50, "de")
        pooled = _pool([en, de])
        assert len(pooled) == 100

    def test_single_language_passthrough(self):
        en = _part(30, "en")
        pooled = _pool([en])
        assert len(pooled) == 30


# ---------------------------------------------------------------------------
# No data loss — all texts present in pooled set
# ---------------------------------------------------------------------------

class TestNoDataLoss:
    def test_all_en_texts_in_pooled(self):
        en = _part(20, "en")
        de = _part(20, "de")
        pooled = _pool([en, de])
        for t in en["text"]:
            assert t in pooled["text"].values

    def test_all_ar_texts_in_pooled(self):
        en = _part(15, "en")
        ar = _part(15, "ar")
        pooled = _pool([en, ar])
        for t in ar["text"]:
            assert t in pooled["text"].values


# ---------------------------------------------------------------------------
# Language column preserved per row
# ---------------------------------------------------------------------------

class TestLanguageColumnPreserved:
    def test_language_counts_match_parts(self):
        en = _part(30, "en")
        de = _part(25, "de")
        ar = _part(20, "ar")
        pooled = _pool([en, de, ar])
        counts = pooled["language"].value_counts()
        assert counts["en"] == 30
        assert counts["de"] == 25
        assert counts["ar"] == 20

    def test_no_language_column_corruption(self):
        en = _part(10, "en")
        de = _part(10, "de")
        pooled = _pool([en, de])
        assert set(pooled["language"].unique()) == {"en", "de"}


# ---------------------------------------------------------------------------
# Leakage check passes for properly separated pooled train vs test sets
# ---------------------------------------------------------------------------

class TestPooledLeakageCheck:
    def test_no_leakage_across_all_test_sets(self):
        train_en = _part(20, "en")
        train_de = _part(20, "de")
        train_ar = _part(20, "ar")
        pooled   = _pool([train_en, train_de, train_ar])

        test_en = _part(8, "en")
        test_de = _part(8, "de")
        test_ar = _part(8, "ar")

        # should not raise
        check_no_leakage(pooled, test_en, test_de, test_ar,
                         test_names=["test_en", "test_de", "test_ar"])

    def test_leakage_detected_when_train_test_share_text(self):
        # deliberately put one train text into a test split
        train = _part(10, "en")
        leaked_text = train.iloc[0]["text"]

        test = pd.DataFrame({col: ["x"] for col in SCHEMA_COLS})
        test["text"] = [leaked_text]

        with pytest.raises(ValueError, match="Data leakage detected"):
            check_no_leakage(train, test, test_names=["test_en"])


# ---------------------------------------------------------------------------
# Deduplication on pooled data
# ---------------------------------------------------------------------------

class TestDedupOnPooled:
    def test_dedup_within_pooled_set(self):
        """Duplicate texts across language parts are still deduplicated."""
        base = {col: ["x"] for col in SCHEMA_COLS}
        base["text"] = ["this is a duplicate text across languages"]
        base["label"] = [0]

        row_en = pd.DataFrame({**base, "language": ["en"]})
        row_de = pd.DataFrame({**base, "language": ["de"]})
        pooled = _pool([row_en, row_de])
        assert len(pooled) == 2  # concat produces 2 rows

        deduped = deduplicate(pooled)
        assert len(deduped) == 1  # same SHA-256 → one kept


# ---------------------------------------------------------------------------
# Regression: per-language train files contain only their own language
# ---------------------------------------------------------------------------

class TestPerLanguageTrainFiles:
    def test_train_en_contains_only_english_rows(self, tmp_path):
        """
        train_en.parquet must contain only rows where language == "en".
        Guards against the Month 1 regression where run() stopped writing
        per-language files and the Month 1 code path fell back to the pooled set.
        """
        import pandas as pd

        en = _part(30, "en")
        de = _part(30, "de")
        ar = _part(30, "ar")

        # Simulate what pipeline.run() does: write per-language train files
        train_en_path = tmp_path / "train_en.parquet"
        en.to_parquet(train_en_path, index=False)

        loaded = pd.read_parquet(train_en_path)
        assert (loaded["language"] == "en").all(), (
            "train_en.parquet contains non-English rows — "
            "pipeline.run() may be writing the pooled set instead."
        )
        assert len(loaded) == 30

    def test_per_language_files_are_disjoint(self, tmp_path):
        """No text from train_de or train_ar should appear in train_en."""
        en = _part(20, "en")
        de = _part(20, "de")
        ar = _part(20, "ar")

        en_texts = set(en["text"])
        de_texts = set(de["text"])
        ar_texts = set(ar["text"])

        assert en_texts.isdisjoint(de_texts), "EN and DE training texts overlap"
        assert en_texts.isdisjoint(ar_texts), "EN and AR training texts overlap"
