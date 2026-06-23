"""
Load MultiTuDe v3 from the local CSV file.

multitude_v3_clean.csv is a single flat file with columns:
  text, label (0/1), multi_label (generator name or "human"),
  split ("train"/"test"), language ("en"/"de"), length, source.

This script filters to English and German, caches each split as a parquet
file under data/raw/multitude_v3/, and validates row counts and generator
counts against the confirmed figures from Macko et al., 2025.

Cached outputs:
  data/raw/multitude_v3/train_en.parquet   (7,954 rows)
  data/raw/multitude_v3/test_en.parquet    (2,384 rows)
  data/raw/multitude_v3/train_de.parquet   (7,951 rows — reserved for Month 3)
  data/raw/multitude_v3/test_de.parquet    (2,388 rows)
"""
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)


def _validate_split(name: str, df: pd.DataFrame,
                    expected_rows: int, expected_generators: int) -> None:
    actual = len(df)
    if actual != expected_rows:
        raise ValueError(
            f"MultiTuDe v3 {name}: expected {expected_rows} rows, got {actual}."
        )
    n_generators = df["generator"].nunique()
    if n_generators != expected_generators:
        raise ValueError(
            f"MultiTuDe v3 {name}: expected {expected_generators} distinct "
            f"generator values, got {n_generators}: {sorted(df['generator'].unique())}"
        )
    logger.info("  %s: %d rows, %d generators ✓", name, actual, n_generators)
    for gen, count in df["generator"].value_counts().items():
        logger.info("    %-30s %d", gen, count)


def load(force: bool = False) -> dict[str, Path]:
    """
    Filter and cache all four MultiTuDe v3 splits.

    Returns a dict mapping split name → absolute Path to the cached parquet.
    Reads from cache on subsequent calls unless force=True.
    """
    cfg = load_config("datasets")["multitude_v3"]
    csv_path = resolve_path(cfg["csv_path"])
    raw_dir = csv_path.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_paths: dict[str, Path] = {
        "train_en": raw_dir / "train_en.parquet",
        "test_en":  raw_dir / "test_en.parquet",
        "train_de": raw_dir / "train_de.parquet",
        "test_de":  raw_dir / "test_de.parquet",
    }

    if all(p.exists() for p in out_paths.values()) and not force:
        logger.info("MultiTuDe v3 splits already cached at %s — skipping.", raw_dir)
        return out_paths

    if not csv_path.exists():
        raise FileNotFoundError(
            f"MultiTuDe v3 CSV not found: {csv_path}\n"
            f"Place multitude_v3_clean.csv under data/raw/multitude_v3/ "
            f"and update multitude_v3.csv_path in configs/datasets.yaml if needed."
        )

    logger.info("Reading %s …", csv_path)
    df = pd.read_csv(csv_path)

    text_col      = cfg["text_col"]
    label_col     = cfg["label_col"]
    generator_col = cfg["generator_col"]
    split_col     = cfg["split_col"]
    language_col  = cfg["language_col"]

    # Rename to canonical names used throughout the pipeline
    df = df.rename(columns={
        text_col:      "text",
        label_col:     "label",
        generator_col: "generator",
        split_col:     "split",
        language_col:  "language",
    })
    df["label"] = df["label"].astype(int)

    expected_rows       = cfg["expected_rows"]
    expected_generators = cfg["expected_generators"]

    split_map = {
        "train_en": ("en", "train"),
        "test_en":  ("en", "test"),
        "train_de": ("de", "train"),
        "test_de":  ("de", "test"),
    }

    for split_name, (lang, split_val) in split_map.items():
        logger.info("Processing %s …", split_name)
        subset = df[(df["language"] == lang) & (df["split"] == split_val)].copy()
        subset = subset.reset_index(drop=True)
        _validate_split(split_name, subset,
                        expected_rows[split_name], expected_generators)
        subset.to_parquet(out_paths[split_name], index=False)

    logger.info("All splits cached under %s", raw_dir)
    return out_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = load()
    for name, path in paths.items():
        print(f"{name}: {path}")
