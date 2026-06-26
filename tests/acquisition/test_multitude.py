"""
Tests for src/acquisition/multitude.py — config-driven language acquisition.

All tests use synthetic in-memory data; no real CSV is read.

Scalability acceptance criterion (from Month 2 PRD):
  Adding a 4th language requires only a config edit, not a code change —
  tested by TestFourthLanguageRequiresNoCodeChange.
"""
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_csv_df() -> pd.DataFrame:
    """Minimal synthetic CSV matching the v3 schema for en, de, ar."""
    rows = []
    gen_names = ["human", "gpt-3.5-turbo-0125", "aya-101",
                 "Mistral-7B-Instruct-v0.2", "v5-Eagle-7B-HF",
                 "vicuna-13b", "opt-iml-max-30b", "Llama-2-70b-chat-hf"]
    for lang in ("en", "de", "ar", "fr"):   # fr is an extra language
        for split in ("train", "test"):
            for i, gen in enumerate(gen_names):
                n = 5 if split == "train" else 2
                for j in range(n):
                    rows.append({
                        "text":        f"{lang}-{split}-{gen}-{j} padded well",
                        "label":       0 if gen == "human" else 1,
                        "multi_label": gen,
                        "split":       split,
                        "language":    lang,
                    })
    return pd.DataFrame(rows)


def _fake_config_3langs() -> dict:
    gen_names = ["human", "gpt-3.5-turbo-0125", "aya-101",
                 "Mistral-7B-Instruct-v0.2", "v5-Eagle-7B-HF",
                 "vicuna-13b", "opt-iml-max-30b", "Llama-2-70b-chat-hf"]
    return {
        "multitude_v3": {
            "csv_path":          "data/raw/multitude_v3/multitude_v3_clean.csv",
            "languages":         ["en", "de", "ar"],
            "train_languages":   ["en", "de", "ar"],
            "expected_rows": {
                "en": {"train": 5 * len(gen_names), "test": 2 * len(gen_names)},
                "de": {"train": 5 * len(gen_names), "test": 2 * len(gen_names)},
                "ar": {"train": 5 * len(gen_names), "test": 2 * len(gen_names)},
            },
            "expected_generators": 8,
            "text_col":      "text",
            "label_col":     "label",
            "generator_col": "multi_label",
            "split_col":     "split",
            "language_col":  "language",
        }
    }


def _fake_config_with_extra_lang(extra: str = "fr") -> dict:
    """Config extended with a 4th language — only config changes, not code."""
    gen_names = ["human", "gpt-3.5-turbo-0125", "aya-101",
                 "Mistral-7B-Instruct-v0.2", "v5-Eagle-7B-HF",
                 "vicuna-13b", "opt-iml-max-30b", "Llama-2-70b-chat-hf"]
    cfg = _fake_config_3langs()
    cfg["multitude_v3"]["languages"].append(extra)
    cfg["multitude_v3"]["train_languages"].append(extra)
    cfg["multitude_v3"]["expected_rows"][extra] = {
        "train": 5 * len(gen_names),
        "test":  2 * len(gen_names),
    }
    return cfg


# ---------------------------------------------------------------------------
# Basic acquisition: all configured splits cached
# ---------------------------------------------------------------------------

class TestLoadProducesExpectedFiles:
    def test_all_language_splits_created(self, tmp_path):
        from src.acquisition.multitude import load

        csv_path = tmp_path / "multitude_v3_clean.csv"
        _fake_csv_df().to_csv(csv_path, index=False)

        cfg = _fake_config_3langs()
        cfg["multitude_v3"]["csv_path"] = str(csv_path)

        with patch("src.acquisition.multitude.load_config", return_value=cfg), \
             patch("src.acquisition.multitude.resolve_path", side_effect=lambda p: tmp_path / Path(p).name):
            paths = load(force=True)

        expected_keys = {f"{s}_{l}" for l in ("en", "de", "ar") for s in ("train", "test")}
        assert set(paths.keys()) == expected_keys
        for key, path in paths.items():
            assert path.exists(), f"{key} not cached"

    def test_cached_parquet_has_correct_columns(self, tmp_path):
        from src.acquisition.multitude import load

        csv_path = tmp_path / "multitude_v3_clean.csv"
        _fake_csv_df().to_csv(csv_path, index=False)
        cfg = _fake_config_3langs()
        cfg["multitude_v3"]["csv_path"] = str(csv_path)

        with patch("src.acquisition.multitude.load_config", return_value=cfg), \
             patch("src.acquisition.multitude.resolve_path", side_effect=lambda p: tmp_path / Path(p).name):
            paths = load(force=True)

        df = pd.read_parquet(paths["train_en"])
        assert "text"      in df.columns
        assert "label"     in df.columns
        assert "generator" in df.columns

    def test_arabic_split_cached_correctly(self, tmp_path):
        from src.acquisition.multitude import load

        csv_path = tmp_path / "multitude_v3_clean.csv"
        _fake_csv_df().to_csv(csv_path, index=False)
        cfg = _fake_config_3langs()
        cfg["multitude_v3"]["csv_path"] = str(csv_path)

        with patch("src.acquisition.multitude.load_config", return_value=cfg), \
             patch("src.acquisition.multitude.resolve_path", side_effect=lambda p: tmp_path / Path(p).name):
            paths = load(force=True)

        df = pd.read_parquet(paths["train_ar"])
        assert (df["language"] == "ar").all()


# ---------------------------------------------------------------------------
# Cache hit: network/disk not read twice
# ---------------------------------------------------------------------------

class TestCacheSkip:
    def test_second_call_does_not_re_read_csv(self, tmp_path):
        from src.acquisition.multitude import load

        csv_path = tmp_path / "multitude_v3_clean.csv"
        _fake_csv_df().to_csv(csv_path, index=False)
        cfg = _fake_config_3langs()
        cfg["multitude_v3"]["csv_path"] = str(csv_path)

        with patch("src.acquisition.multitude.load_config", return_value=cfg), \
             patch("src.acquisition.multitude.resolve_path", side_effect=lambda p: tmp_path / Path(p).name):
            load(force=True)

        with patch("src.acquisition.multitude.load_config", return_value=cfg), \
             patch("src.acquisition.multitude.resolve_path", side_effect=lambda p: tmp_path / Path(p).name), \
             patch("pandas.read_csv") as mock_read:
            load(force=False)
            mock_read.assert_not_called()


# ---------------------------------------------------------------------------
# Scalability: adding a 4th language requires only a config edit
# ---------------------------------------------------------------------------

class TestFourthLanguageRequiresNoCodeChange:
    def test_fourth_language_loaded_with_config_only_change(self, tmp_path):
        """
        French ("fr") is not in the Month 2 config. Adding it to the config
        dict — with no code changes — must produce cached train_fr / test_fr.
        """
        from src.acquisition.multitude import load

        csv_path = tmp_path / "multitude_v3_clean.csv"
        _fake_csv_df().to_csv(csv_path, index=False)

        cfg = _fake_config_with_extra_lang("fr")
        cfg["multitude_v3"]["csv_path"] = str(csv_path)

        with patch("src.acquisition.multitude.load_config", return_value=cfg), \
             patch("src.acquisition.multitude.resolve_path", side_effect=lambda p: tmp_path / Path(p).name):
            paths = load(force=True)

        assert "train_fr" in paths
        assert "test_fr"  in paths
        assert paths["train_fr"].exists()
        df = pd.read_parquet(paths["train_fr"])
        assert (df["language"] == "fr").all()


# ---------------------------------------------------------------------------
# Validation: wrong row count / generator count
# ---------------------------------------------------------------------------

class TestValidation:
    def test_wrong_row_count_raises(self, tmp_path):
        from src.acquisition.multitude import _validate_split

        gen_names = ["human", "gpt4", "aya", "mistral", "eagle", "vicuna", "opt", "llama"]
        df = pd.DataFrame({
            "text":      [f"t{i}" for i in range(5)],
            "label":     [0] * 5,
            "generator": gen_names[:5],
        })
        with pytest.raises(ValueError, match="expected 10 rows"):
            _validate_split("train_en", df, expected_rows=10, expected_generators=5)

    def test_wrong_generator_count_raises(self, tmp_path):
        from src.acquisition.multitude import _validate_split

        gen_names = ["human", "gpt4", "aya"]
        df = pd.DataFrame({
            "text":      [f"text sample {i}" for i in range(3)],
            "label":     [0] * 3,
            "generator": gen_names,
        })
        with pytest.raises(ValueError, match="expected 8 generators"):
            _validate_split("train_en", df, expected_rows=3, expected_generators=8)

    def test_missing_csv_raises(self, tmp_path):
        from src.acquisition.multitude import load

        cfg = _fake_config_3langs()
        missing = str(tmp_path / "does_not_exist.csv")
        cfg["multitude_v3"]["csv_path"] = missing

        with patch("src.acquisition.multitude.load_config", return_value=cfg), \
             patch("src.acquisition.multitude.resolve_path", side_effect=lambda p: Path(p)):
            with pytest.raises(FileNotFoundError):
                load(force=True)
