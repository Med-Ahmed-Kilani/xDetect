"""
End-to-end smoke test: runs the full pipeline on tiny synthetic data
to verify wiring before any real dataset downloads.

No real data, no model downloads — all stubs use in-process mocks.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


_COUNTER = 0


def _synthetic_df(n: int = 40, language: str = "en") -> pd.DataFrame:
    global _COUNTER
    rng = np.random.default_rng(42)
    offset = _COUNTER
    _COUNTER += n
    return pd.DataFrame({
        "text": [f"synthetic text sample number {offset + i} long enough" for i in range(n)],
        "label": rng.integers(0, 2, n).tolist(),
        "language": [language] * n,
        "source_dataset": ["multitude_v3"] * n,
        "domain": ["news"] * n,
        "generator": rng.choice(["human", "gpt4"], n).tolist(),
    })


class TestSmokePipeline:
    """Verify the preprocessing + deduplication + leakage-check chain."""

    def test_pipeline_wiring(self, tmp_path):
        from src.preprocessing.pipeline import check_no_leakage, deduplicate, clean_text

        train = _synthetic_df(40, "en")
        test_en = _synthetic_df(10, "en")
        test_de = _synthetic_df(10, "de")

        # All texts are unique by construction — leakage check must pass
        check_no_leakage(train, test_en, test_de, test_names=["test_en", "test_de"])

        # Deduplication should be a no-op on unique data
        deduped = deduplicate(train)
        assert len(deduped) == len(train)

    def test_dedup_removes_synthetic_duplicates(self):
        from src.preprocessing.pipeline import deduplicate, SCHEMA_COLS

        texts = ["this text is duplicated exactly"] * 5 + [
            f"unique text entry number {i}" for i in range(5)
        ]
        df = pd.DataFrame({
            "text": texts,
            "label": [0] * 10,
            "language": ["en"] * 10,
            "source_dataset": ["test"] * 10,
            "domain": ["qa"] * 10,
            "generator": ["human"] * 10,
        })
        result = deduplicate(df)
        assert len(result) == 6  # 1 unique dup + 5 unique others

    def test_leakage_check_raises_on_overlap(self):
        from src.preprocessing.pipeline import check_no_leakage

        shared_text = "this text appears in both train and test sets really"
        train = _synthetic_df(5, "en")
        train.loc[0, "text"] = shared_text
        test  = _synthetic_df(5, "en")
        test.loc[0, "text"] = shared_text

        with pytest.raises(ValueError, match="leakage"):
            check_no_leakage(train, test, test_names=["test"])


class TestSmokeEval:
    """Verify eval harness output schema on synthetic scored data."""

    def test_eval_split_schema(self):
        from src.eval.harness import _eval_split

        rng = np.random.default_rng(7)
        n = 20
        df = pd.DataFrame({
            "text": [f"text {i}" for i in range(n)],
            "label": rng.integers(0, 2, n).tolist(),
            "language": ["en"] * n,
            "source_dataset": ["multitude_v3"] * n,
            "domain": ["news"] * n,
            "generator": rng.choice(["human", "gpt4"], n).tolist(),
            "pred": rng.integers(0, 2, n).tolist(),
            "proba": rng.random(n).tolist(),
        })

        result = _eval_split(df, "pred", "proba", "test_en", "supervised")
        assert "overall" in result
        assert "per_generator" in result
        assert result["overall"]["n"] == n
        assert isinstance(result["overall"]["accuracy"], float)
        assert isinstance(result["overall"]["f1"], float)

    def test_report_json_serializable(self):
        from src.eval.harness import _eval_split

        rng = np.random.default_rng(9)
        n = 10
        df = pd.DataFrame({
            "text": [f"text {i}" for i in range(n)],
            "label": rng.integers(0, 2, n).tolist(),
            "language": ["en"] * n,
            "source_dataset": ["multitude_v3"] * n,
            "domain": ["news"] * n,
            "generator": ["human"] * (n // 2) + ["gpt4"] * (n - n // 2),
            "pred": rng.integers(0, 2, n).tolist(),
            "proba": rng.random(n).tolist(),
        })
        result = _eval_split(df, "pred", "proba", "test_en", "supervised")
        # Must be JSON-serializable (no numpy scalars leaking)
        json.dumps(result)
