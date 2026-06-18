"""
Unified preprocessing pipeline.

Reads raw parquet files from data/raw/, applies cleaning, schema unification,
and deduplication, then writes:

  data/processed/train_en.parquet
  data/processed/test_en.parquet
  data/processed/test_de.parquet

A leakage check confirms zero text-hash overlap between the training set and
either test set before writing the test outputs.
"""
import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

SCHEMA_COLS = ["text", "label", "language", "source_dataset", "domain", "generator"]


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _strip_control_chars(text: str) -> str:
    return _CTRL_RE.sub("", text)


def clean_text(text: str) -> Optional[str]:
    """
    Normalize and clean a single text string.

    Returns None if the text is empty or too short after cleaning.
    """
    if not isinstance(text, str):
        return None
    text = _normalize_unicode(text)
    text = _strip_control_chars(text)
    text = text.strip()
    # Discard texts shorter than 10 characters — too short to be meaningful
    if len(text) < 10:
        return None
    return text


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# Schema unification
# ---------------------------------------------------------------------------

def _unify_multitude(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Map MultiTuDe raw columns to the unified schema."""
    # Determine language from split name
    language = "de" if split == "test_de" else "en"

    # MultiTuDe column names may vary; handle common variants
    col_map = {}
    for candidate in ("text", "content", "article"):
        if candidate in df.columns:
            col_map[candidate] = "text"
            break
    for candidate in ("label", "class", "is_generated"):
        if candidate in df.columns and candidate != "label":
            col_map[candidate] = "label"
            break
    for candidate in ("model", "generator", "source_model"):
        if candidate in df.columns:
            col_map[candidate] = "generator"
            break
    if col_map:
        df = df.rename(columns=col_map)

    # Ensure label is 0/1 int
    if "label" in df.columns and df["label"].dtype == object:
        label_map = {"human": 0, "machine": 1, "generated": 1, "ai": 1}
        df["label"] = df["label"].str.lower().map(label_map).fillna(df["label"])
    df["label"] = df["label"].astype(int)

    df["language"] = language
    df["source_dataset"] = "multitude"
    df["domain"] = "news"
    if "generator" not in df.columns:
        df["generator"] = "unknown"

    return df[SCHEMA_COLS]


def _unify_hc3(df: pd.DataFrame) -> pd.DataFrame:
    """Map HC3 raw columns to the unified schema."""
    df = df.copy()
    df["language"] = "en"
    df["source_dataset"] = "hc3"
    df["domain"] = "qa"
    # generator column already present from acquisition
    return df[SCHEMA_COLS]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate texts (by SHA-256 hash)."""
    before = len(df)
    df = df.copy()
    df["_hash"] = df["text"].apply(text_hash)
    df = df.drop_duplicates(subset=["_hash"])
    df = df.drop(columns=["_hash"])
    removed = before - len(df)
    if removed:
        logger.info("Deduplication removed %d duplicate rows.", removed)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------

def check_no_leakage(train_df: pd.DataFrame, *test_dfs: pd.DataFrame,
                     test_names: list[str] | None = None) -> None:
    """
    Assert that no text in any test set appears in the training set.

    Raises ValueError if any overlap is detected.
    """
    train_hashes = set(train_df["text"].apply(text_hash))
    names = test_names or [f"test_{i}" for i in range(len(test_dfs))]
    for name, test_df in zip(names, test_dfs):
        test_hashes = set(test_df["text"].apply(text_hash))
        overlap = train_hashes & test_hashes
        if overlap:
            raise ValueError(
                f"Data leakage detected: {len(overlap)} texts appear in both "
                f"train and {name}."
            )
        logger.info("Leakage check %s: no overlap ✓", name)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _clean_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Apply text cleaning to an already-unified DataFrame."""
    before = len(df)
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    df = df.dropna(subset=["text"])
    dropped = before - len(df)
    if dropped:
        logger.info("  [%s] Dropped %d rows after text cleaning.", source, dropped)
    return df


def _load_and_clean(path: Path, source: str, split: str) -> pd.DataFrame:
    """Load a MultiTuDe parquet file, unify schema, and clean text."""
    logger.info("Loading %s from %s …", split, path)
    df = pd.read_parquet(path)
    df = _unify_multitude(df, split)
    return _clean_df(df, source)


def run(force: bool = False) -> dict[str, Path]:
    """
    Run the full preprocessing pipeline.

    Returns paths to the three processed parquet files.
    """
    from src.acquisition.hc3 import load as load_hc3

    cfg_ds = load_config("datasets")
    raw_multitude = resolve_path(cfg_ds["multitude"]["raw_dir"])
    processed_cfg = cfg_ds["processed"]

    out_train = resolve_path(processed_cfg["train_en"])
    out_test_en = resolve_path(processed_cfg["test_en"])
    out_test_de = resolve_path(processed_cfg["test_de"])

    if all(p.exists() for p in [out_train, out_test_en, out_test_de]) and not force:
        logger.info("Processed files already exist — skipping preprocessing.")
        return {"train_en": out_train, "test_en": out_test_en, "test_de": out_test_de}

    out_train.parent.mkdir(parents=True, exist_ok=True)

    # Load and clean individual sources
    mt_train = _load_and_clean(raw_multitude / "train_en.parquet", "multitude", "train_en")
    mt_test_en = _load_and_clean(raw_multitude / "test_en.parquet", "multitude", "test_en")
    mt_test_de = _load_and_clean(raw_multitude / "test_de.parquet", "multitude", "test_de")
    hc3_en = _clean_df(load_hc3(), "hc3")

    # Merge training set
    train_df = pd.concat([mt_train, hc3_en], ignore_index=True)
    train_df = deduplicate(train_df)

    # Deduplicate test sets independently
    test_en_df = deduplicate(mt_test_en)
    test_de_df = deduplicate(mt_test_de)

    # Leakage check — must pass before writing
    check_no_leakage(train_df, test_en_df, test_de_df,
                     test_names=["test_en", "test_de"])

    logger.info("Writing processed files …")
    train_df.to_parquet(out_train, index=False)
    test_en_df.to_parquet(out_test_en, index=False)
    test_de_df.to_parquet(out_test_de, index=False)

    logger.info("train_en: %d rows", len(train_df))
    logger.info("test_en:  %d rows", len(test_en_df))
    logger.info("test_de:  %d rows", len(test_de_df))

    return {"train_en": out_train, "test_en": out_test_en, "test_de": out_test_de}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = run()
    for name, path in paths.items():
        print(f"{name}: {path}")
