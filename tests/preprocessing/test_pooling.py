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
# Regression: per-language train files produced by pipeline.run()
# ---------------------------------------------------------------------------
#
# These tests call pipeline.run() against a small synthetic multi-language
# dataset and assert on the actual parquet files it writes to disk.
# They catch regressions in the real pipeline logic — not manually-constructed
# DataFrames that bypass run() entirely.

from unittest.mock import patch
from pathlib import Path as _Path


def _make_raw_split(raw_dir: _Path, split: str, lang: str, n: int) -> None:
    """Write a minimal raw parquet as multitude.load() would produce it."""
    global _LID
    offset = _LID
    _LID += n
    df = pd.DataFrame({
        "text":      [f"{lang}-{split}-raw-{offset + i} long enough to survive cleaning" for i in range(n)],
        "label":     [i % 2 for i in range(n)],
        "generator": [("human" if i % 2 == 0 else "gpt4") for i in range(n)],
        "language":  [lang] * n,
        "split":     [split] * n,
    })
    df.to_parquet(raw_dir / f"{split}_{lang}.parquet", index=False)


def _pipeline_config(tmp_path: _Path, languages: list[str]) -> dict:
    """Return a load_config-shaped dict pointing all paths at tmp_path."""
    raw_dir  = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    return {
        "multitude_v3": {
            "csv_path":        str(raw_dir / "multitude_v3_clean.csv"),
            "languages":       languages,
            "train_languages": languages,
        },
        "processed": {
            "train_pooled":   str(proc_dir / "train_pooled.parquet"),
            "train_template": str(proc_dir / "train_{lang}.parquet"),
            "test_template":  str(proc_dir / "test_{lang}.parquet"),
        },
    }


def _run_pipeline(tmp_path: _Path, languages: list[str]) -> None:
    """
    Prepare raw splits under tmp_path, patch config/resolve_path, then
    call pipeline.run(force=True) so it exercises the real write logic.
    """
    from src.preprocessing.pipeline import run

    raw_dir  = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir()
    proc_dir.mkdir()

    for lang in languages:
        _make_raw_split(raw_dir, "train", lang, n=20)
        _make_raw_split(raw_dir, "test",  lang, n=8)

    cfg = _pipeline_config(tmp_path, languages)

    with patch("src.preprocessing.pipeline.load_config", return_value=cfg), \
         patch("src.preprocessing.pipeline.resolve_path", side_effect=_Path):
        run(force=True)


class TestPerLanguageTrainFiles:
    def test_train_en_written_by_run(self, tmp_path):
        """pipeline.run() must create train_en.parquet on disk."""
        _run_pipeline(tmp_path, ["en", "de", "ar"])
        out = tmp_path / "processed" / "train_en.parquet"
        assert out.exists(), "train_en.parquet was not written by pipeline.run()"

    def test_train_en_contains_only_english_rows(self, tmp_path):
        """
        train_en.parquet produced by pipeline.run() must contain only rows
        where language == 'en'. Catches the regression where run() wrote
        train_pooled (EN+DE+AR) instead of the per-language file.
        """
        _run_pipeline(tmp_path, ["en", "de", "ar"])
        out = pd.read_parquet(tmp_path / "processed" / "train_en.parquet")
        assert (out["language"] == "en").all(), (
            "train_en.parquet contains non-English rows — "
            "pipeline.run() may be writing the pooled set instead."
        )

    def test_per_language_files_are_disjoint(self, tmp_path):
        """Texts in train_en must not appear in train_de or train_ar."""
        _run_pipeline(tmp_path, ["en", "de", "ar"])
        proc = tmp_path / "processed"
        en_texts = set(pd.read_parquet(proc / "train_en.parquet")["text"])
        de_texts = set(pd.read_parquet(proc / "train_de.parquet")["text"])
        ar_texts = set(pd.read_parquet(proc / "train_ar.parquet")["text"])
        assert en_texts.isdisjoint(de_texts), "train_en and train_de share texts"
        assert en_texts.isdisjoint(ar_texts), "train_en and train_ar share texts"

    def test_train_en_row_count_matches_raw_input(self, tmp_path):
        """train_en.parquet row count must equal the raw EN train split (20 rows)."""
        _run_pipeline(tmp_path, ["en", "de", "ar"])
        out = pd.read_parquet(tmp_path / "processed" / "train_en.parquet")
        assert len(out) == 20

    def test_all_per_language_files_written(self, tmp_path):
        """run() must write train_{lang}.parquet for every configured language."""
        _run_pipeline(tmp_path, ["en", "de", "ar"])
        proc = tmp_path / "processed"
        for lang in ("en", "de", "ar"):
            assert (proc / f"train_{lang}.parquet").exists(), (
                f"train_{lang}.parquet missing from pipeline output"
            )
