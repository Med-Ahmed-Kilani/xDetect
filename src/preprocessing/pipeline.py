"""
Unified preprocessing pipeline — MultiTuDe v3 only.

Reads cached parquet splits from data/raw/multitude_v3/, applies cleaning,
schema unification, and deduplication, then writes:

  data/processed/train_en.parquet  — English training set (EN split only)
  data/processed/test_en.parquet   — English test set (frozen)
  data/processed/test_de.parquet   — German test set (frozen, zero-shot eval)

Note: the German training split (train_de) exists in v3 but is intentionally
not used here — it is reserved for the Month 3 adapter experiments.

A leakage check confirms zero text-hash overlap between the training set and
either test set before writing the outputs.
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
# Text cleaning
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
    if len(text) < 10:
        return None
    return text


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# Schema unification
# ---------------------------------------------------------------------------

def _unify_multitude_v3(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Map a cached MultiTuDe v3 parquet split to the unified pipeline schema.

    The raw parquet already has 'text', 'label', 'generator', 'language'
    columns from the acquisition step — we just add the fixed metadata fields.
    """
    df = df.copy()
    df["language"]       = language
    df["source_dataset"] = "multitude_v3"
    df["domain"]         = "news"
    df["label"]          = df["label"].astype(int)
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
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_df(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Apply text cleaning and drop rows that don't survive it."""
    before = len(df)
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    df = df.dropna(subset=["text"])
    dropped = before - len(df)
    if dropped:
        logger.info("  [%s] Dropped %d rows after text cleaning.", split_name, dropped)
    return df


def _log_generator_counts(split_name: str, df: pd.DataFrame) -> None:
    logger.info("%s generator breakdown:", split_name)
    for gen, count in df["generator"].value_counts().items():
        logger.info("  %-30s %d", gen, count)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run(force: bool = False) -> dict[str, Path]:
    """
    Run the full preprocessing pipeline.

    Returns paths to the three processed parquet files.
    """
    cfg_ds = load_config("datasets")
    raw_dir = resolve_path(cfg_ds["multitude_v3"]["csv_path"]).parent
    processed_cfg = cfg_ds["processed"]

    out_train  = resolve_path(processed_cfg["train_en"])
    out_test_en = resolve_path(processed_cfg["test_en"])
    out_test_de = resolve_path(processed_cfg["test_de"])

    if all(p.exists() for p in [out_train, out_test_en, out_test_de]) and not force:
        logger.info("Processed files already exist — skipping preprocessing.")
        return {"train_en": out_train, "test_en": out_test_en, "test_de": out_test_de}

    out_train.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MultiTuDe v3 cached splits …")
    train_df   = _unify_multitude_v3(pd.read_parquet(raw_dir / "train_en.parquet"), "en")
    test_en_df = _unify_multitude_v3(pd.read_parquet(raw_dir / "test_en.parquet"),  "en")
    test_de_df = _unify_multitude_v3(pd.read_parquet(raw_dir / "test_de.parquet"),  "de")

    train_df   = _clean_df(deduplicate(train_df),   "train_en")
    test_en_df = _clean_df(deduplicate(test_en_df), "test_en")
    test_de_df = _clean_df(deduplicate(test_de_df), "test_de")

    check_no_leakage(train_df, test_en_df, test_de_df,
                     test_names=["test_en", "test_de"])

    _log_generator_counts("train_en", train_df)
    _log_generator_counts("test_en",  test_en_df)
    _log_generator_counts("test_de",  test_de_df)

    logger.info("Writing processed files …")
    train_df.to_parquet(out_train,    index=False)
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
