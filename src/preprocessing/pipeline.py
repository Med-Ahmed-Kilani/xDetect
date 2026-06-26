"""
Unified preprocessing pipeline — MultiTuDe v3, config-driven languages.

Reads cached per-language parquet splits from data/raw/multitude_v3/,
applies cleaning, schema unification, and deduplication, then writes:

  data/processed/train_pooled.parquet  — pooled training set (all train_languages)
  data/processed/test_{lang}.parquet   — one frozen test file per language

Languages to acquire and languages to pool in training are both driven by
configs/datasets.yaml (multitude_v3.languages / train_languages).

Note: German and Arabic training splits exist in v3 but are included here
for Month 2's backbone comparison. The German/Arabic data was intentionally
withheld from training in Month 1 to preserve the zero-shot transfer baseline.
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
    """Add fixed metadata fields to a cached v3 parquet split."""
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


def _load_split(raw_dir: Path, split: str, lang: str) -> pd.DataFrame:
    path = raw_dir / f"{split}_{lang}.parquet"
    logger.info("Loading %s_%s from %s …", split, lang, path)
    df = pd.read_parquet(path)
    return _unify_multitude_v3(df, lang)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run(force: bool = False) -> dict[str, Path]:
    """
    Run the full preprocessing pipeline.

    Pools training data for all train_languages and writes one test file per
    language. All language/path decisions are driven by configs/datasets.yaml.

    Returns a dict mapping split key → Path.
    """
    cfg_ds = load_config("datasets")["multitude_v3"]
    raw_dir       = resolve_path(cfg_ds["csv_path"]).parent
    languages     = cfg_ds["languages"]
    train_langs   = cfg_ds["train_languages"]
    processed_cfg = load_config("datasets")["processed"]

    out_train = resolve_path(processed_cfg["train_pooled"])
    out_tests = {
        lang: resolve_path(processed_cfg["test_template"].replace("{lang}", lang))
        for lang in languages
    }
    all_outputs = [out_train, *out_tests.values()]

    if all(p.exists() for p in all_outputs) and not force:
        logger.info("Processed files already exist — skipping preprocessing.")
        return {"train_pooled": out_train, **{f"test_{l}": p for l, p in out_tests.items()}}

    out_train.parent.mkdir(parents=True, exist_ok=True)

    # Load and clean test sets for all languages
    test_dfs: dict[str, pd.DataFrame] = {}
    for lang in languages:
        df = _clean_df(deduplicate(_load_split(raw_dir, "test", lang)), f"test_{lang}")
        _log_generator_counts(f"test_{lang}", df)
        test_dfs[lang] = df

    # Load, clean, and pool training sets
    train_parts: list[pd.DataFrame] = []
    for lang in train_langs:
        df = _clean_df(deduplicate(_load_split(raw_dir, "train", lang)), f"train_{lang}")
        _log_generator_counts(f"train_{lang}", df)
        train_parts.append(df)

    train_df = pd.concat(train_parts, ignore_index=True)
    expected_pool = sum(len(p) for p in train_parts)
    assert len(train_df) == expected_pool, (
        f"Pooled row count {len(train_df)} != sum of parts {expected_pool}"
    )
    logger.info("Pooled training set: %d rows across %s", len(train_df), train_langs)

    # Leakage check — pooled train vs every test
    check_no_leakage(
        train_df,
        *[test_dfs[lang] for lang in languages],
        test_names=[f"test_{lang}" for lang in languages],
    )

    logger.info("Writing processed files …")
    train_df.to_parquet(out_train, index=False)
    for lang, df in test_dfs.items():
        df.to_parquet(out_tests[lang], index=False)

    logger.info("train_pooled: %d rows", len(train_df))
    for lang, df in test_dfs.items():
        logger.info("test_%s:     %d rows", lang, len(df))

    return {"train_pooled": out_train, **{f"test_{l}": p for l, p in out_tests.items()}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = run()
    for name, path in paths.items():
        print(f"{name}: {path}")
