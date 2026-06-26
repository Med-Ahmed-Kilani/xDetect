"""
Load MultiTuDe v3 splits from the local CSV file.

multitude_v3_clean.csv is a single flat file. This script filters to every
language listed in configs/datasets.yaml (multitude_v3.languages) and caches
each train/test split as a parquet file.

Adding a new language requires only adding its code to `languages` and its
expected row counts to `expected_rows` in configs/datasets.yaml — no code change.

Cached output layout (one file per language per split):
  data/raw/multitude_v3/train_{lang}.parquet
  data/raw/multitude_v3/test_{lang}.parquet
"""
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)


def _validate_split(key: str, df: pd.DataFrame,
                    expected_rows: int, expected_generators: int) -> None:
    actual = len(df)
    if actual != expected_rows:
        raise ValueError(
            f"MultiTuDe v3 {key}: expected {expected_rows} rows, got {actual}."
        )
    n_gen = df["generator"].nunique()
    if n_gen != expected_generators:
        raise ValueError(
            f"MultiTuDe v3 {key}: expected {expected_generators} generators, "
            f"got {n_gen}: {sorted(df['generator'].unique())}"
        )
    logger.info("  %s: %d rows, %d generators ✓", key, actual, n_gen)
    for gen, count in df["generator"].value_counts().items():
        logger.info("    %-30s %d", gen, count)


def load(force: bool = False) -> dict[str, Path]:
    """
    Filter and cache all configured language splits from the v3 CSV.

    Returns a dict mapping "{split}_{lang}" → absolute Path to cached parquet.
    Reads from cache on subsequent calls unless force=True.

    Adding a new language: add its code to `languages` and its expected row
    counts to `expected_rows` in configs/datasets.yaml — this function needs
    no changes.
    """
    cfg = load_config("datasets")["multitude_v3"]
    csv_path = resolve_path(cfg["csv_path"])
    raw_dir = csv_path.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    languages         = cfg["languages"]
    expected_rows_cfg = cfg["expected_rows"]
    expected_generators = cfg["expected_generators"]

    out_paths: dict[str, Path] = {
        f"{split}_{lang}": raw_dir / f"{split}_{lang}.parquet"
        for lang in languages
        for split in ("train", "test")
    }

    if all(p.exists() for p in out_paths.values()) and not force:
        logger.info("MultiTuDe v3 splits already cached at %s — skipping.", raw_dir)
        return out_paths

    if not csv_path.exists():
        raise FileNotFoundError(
            f"MultiTuDe v3 CSV not found: {csv_path}\n"
            f"Place multitude_v3_clean.csv under data/raw/multitude_v3/."
        )

    logger.info("Reading %s …", csv_path)
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        cfg["text_col"]:      "text",
        cfg["label_col"]:     "label",
        cfg["generator_col"]: "generator",
        cfg["split_col"]:     "split",
        cfg["language_col"]:  "language",
    })
    df["label"] = df["label"].astype(int)

    for lang in languages:
        for split in ("train", "test"):
            key = f"{split}_{lang}"
            logger.info("Processing %s …", key)
            subset = df[
                (df["language"] == lang) & (df["split"] == split)
            ].copy().reset_index(drop=True)

            _validate_split(
                key, subset,
                expected_rows_cfg[lang][split],
                expected_generators,
            )
            subset.to_parquet(out_paths[key], index=False)

    logger.info("All splits cached under %s", raw_dir)
    return out_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = load()
    for name, path in paths.items():
        print(f"{name}: {path}")
