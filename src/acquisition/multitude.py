"""
Load MultiTuDe splits from a locally downloaded Zenodo archive.

MultiTuDe is distributed via Zenodo (not HuggingFace). After your Zenodo
access is approved, download the archive and extract it. Then set the path
in configs/datasets.yaml under multitude.zenodo_dir.

Expected layout inside the extracted archive (adjust zenodo_dir if yours differs):
  <zenodo_dir>/
    en/
      train.jsonl   (or .csv / .parquet — see file_format in config)
      test.jsonl
    de/
      test.jsonl

Outputs (cached parquet files):
  data/raw/multitude/train_en.parquet
  data/raw/multitude/test_en.parquet
  data/raw/multitude/test_de.parquet
"""
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

ROW_COUNT_TOLERANCE = 0.05


def _sha256_of_texts(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in sorted(texts[:1000]):
        h.update(t.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _validate_row_count(split_name: str, df: pd.DataFrame, expected: int) -> None:
    actual = len(df)
    delta = abs(actual - expected) / expected
    if delta > ROW_COUNT_TOLERANCE:
        raise ValueError(
            f"MultiTuDe {split_name}: expected ~{expected} rows, got {actual} "
            f"(delta={delta:.1%} exceeds {ROW_COUNT_TOLERANCE:.0%} tolerance)"
        )
    logger.info("  %s: %d rows (expected ~%d) ✓", split_name, actual, expected)


def _read_file(path: Path) -> pd.DataFrame:
    """Read a data file based on its extension."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    elif suffix == ".csv":
        return pd.read_csv(path)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. "
                         f"Update multitude.file_format in configs/datasets.yaml.")


def _find_split_file(base_dir: Path, candidates: list[str]) -> Path:
    """Find the first existing file from a list of candidate names."""
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find any of {candidates} in {base_dir}. "
        f"Check multitude.zenodo_dir in configs/datasets.yaml."
    )


def load(force: bool = False) -> dict[str, Path]:
    """
    Load MultiTuDe from the local Zenodo archive and cache as parquet.

    Parameters
    ----------
    force : re-process even if cached parquet files already exist.

    Returns
    -------
    dict mapping split name → absolute Path to cached parquet.
    """
    cfg = load_config("datasets")["multitude"]
    raw_dir = resolve_path(cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_paths: dict[str, Path] = {
        "train_en": raw_dir / "train_en.parquet",
        "test_en":  raw_dir / "test_en.parquet",
        "test_de":  raw_dir / "test_de.parquet",
    }

    if all(p.exists() for p in out_paths.values()) and not force:
        logger.info("MultiTuDe already cached at %s — skipping.", raw_dir)
        return out_paths

    zenodo_dir = resolve_path(cfg["zenodo_dir"])
    if not zenodo_dir.exists():
        raise FileNotFoundError(
            f"MultiTuDe zenodo_dir not found: {zenodo_dir}\n"
            f"Download the archive from Zenodo and set multitude.zenodo_dir "
            f"in configs/datasets.yaml to the extracted folder path."
        )

    expected_rows = cfg["expected_rows"]
    file_candidates = cfg.get("file_candidates", ["train.jsonl", "train.csv", "train.parquet"])
    text_col = cfg["text_col"]
    label_col = cfg["label_col"]
    generator_col = cfg["generator_col"]

    split_dirs = {
        "train_en": (zenodo_dir / cfg["splits"]["train_en_dir"],
                     cfg["splits"]["train_en_file"]),
        "test_en":  (zenodo_dir / cfg["splits"]["test_en_dir"],
                     cfg["splits"]["test_en_file"]),
        "test_de":  (zenodo_dir / cfg["splits"]["test_de_dir"],
                     cfg["splits"]["test_de_file"]),
    }

    checksums: dict = {}
    for split_name, (split_dir, filename) in split_dirs.items():
        logger.info("Loading %s from %s …", split_name, split_dir / filename)
        src = split_dir / filename
        if not src.exists():
            raise FileNotFoundError(
                f"Expected {src} — check multitude.splits in configs/datasets.yaml."
            )
        df = _read_file(src)

        # Rename columns to standard names using config mapping
        rename = {}
        for cfg_key, standard in [
            (text_col, "text"),
            (label_col, "label"),
            (generator_col, "generator"),
        ]:
            if cfg_key in df.columns and cfg_key != standard:
                rename[cfg_key] = standard
        if rename:
            df = df.rename(columns=rename)

        _validate_row_count(split_name, df, expected_rows[split_name])

        df.to_parquet(out_paths[split_name], index=False)
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
    paths = load()
    for name, path in paths.items():
        print(f"{name}: {path}")
