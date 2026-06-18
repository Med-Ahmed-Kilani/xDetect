"""
Download and cache MultiTuDe splits needed for Month 1.

Fetches:
  - English train split  → data/raw/multitude/train_en.parquet
  - English test split   → data/raw/multitude/test_en.parquet
  - German test split    → data/raw/multitude/test_de.parquet

Row counts are validated against published figures on completion.
"""
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

# Published row counts from Macko et al., 2023
EXPECTED_ROWS = {
    "train_en": 26969,
    "test_en": 2491,
    "test_de": 2685,
}

# Tolerance for row-count validation (±5%)
ROW_COUNT_TOLERANCE = 0.05


def _sha256_of_texts(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in sorted(texts[:1000]):  # fingerprint first 1k for speed
        h.update(t.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _validate_row_count(split_name: str, df: pd.DataFrame) -> None:
    expected = EXPECTED_ROWS[split_name]
    actual = len(df)
    delta = abs(actual - expected) / expected
    if delta > ROW_COUNT_TOLERANCE:
        raise ValueError(
            f"MultiTuDe {split_name}: expected ~{expected} rows, got {actual} "
            f"(delta={delta:.1%} exceeds {ROW_COUNT_TOLERANCE:.0%} tolerance)"
        )
    logger.info("  %s: %d rows (expected ~%d) ✓", split_name, actual, expected)


def _hf_split_to_df(dataset, cfg: dict) -> pd.DataFrame:
    """Convert a HuggingFace dataset split to a raw DataFrame with known columns."""
    df = dataset.to_pandas()
    # Normalise column names to what the config declares
    rename = {}
    for key in ("text_col", "label_col", "generator_col", "language_col"):
        src = cfg.get(key)
        if src and src in df.columns:
            rename[src] = key.replace("_col", "")
    if rename:
        df = df.rename(columns=rename)
    return df


def download(force: bool = False) -> dict[str, Path]:
    """
    Download MultiTuDe and return paths to the cached parquet files.

    Parameters
    ----------
    force : re-download even if cached files already exist.

    Returns
    -------
    dict mapping split name → absolute Path.
    """
    cfg = load_config("datasets")["multitude"]
    raw_dir = resolve_path(cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_paths: dict[str, Path] = {
        "train_en": raw_dir / "train_en.parquet",
        "test_en": raw_dir / "test_en.parquet",
        "test_de": raw_dir / "test_de.parquet",
    }

    all_cached = all(p.exists() for p in out_paths.values())
    if all_cached and not force:
        logger.info("MultiTuDe already cached at %s — skipping download.", raw_dir)
        return out_paths

    logger.info("Downloading MultiTuDe from HuggingFace (%s) …", cfg["hf_id"])

    # MultiTuDe ships with separate split names per language on HF.
    # Verified structure: 'train' (EN train), 'test' (EN test), 'test_de' (DE test).
    dataset = load_dataset(cfg["hf_id"], trust_remote_code=True)

    split_map = {
        "train_en": cfg["splits"]["train_en"],
        "test_en": cfg["splits"]["test_en"],
        "test_de": cfg["splits"]["test_de"],
    }

    checksums: dict[str, str] = {}
    for split_name, hf_split in split_map.items():
        logger.info("Processing split: %s (HF key=%s)", split_name, hf_split)
        ds_split = dataset[hf_split]
        df = _hf_split_to_df(ds_split, cfg)
        _validate_row_count(split_name, df)

        out_path = out_paths[split_name]
        df.to_parquet(out_path, index=False)
        checksums[split_name] = {
            "rows": len(df),
            "text_hash": _sha256_of_texts(df["text"].tolist()) if "text" in df else "",
        }

    log_path = raw_dir / "checksums.json"
    with open(log_path, "w") as f:
        json.dump(checksums, f, indent=2)
    logger.info("Checksums written to %s", log_path)

    return out_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = download()
    for name, path in paths.items():
        print(f"{name}: {path}")
