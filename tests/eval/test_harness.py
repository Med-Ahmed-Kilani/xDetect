"""
Integration tests for the evaluation harness output schema and determinism.

These tests use synthetic data — no real model inference is run.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.eval.harness import _eval_split, _flag_suspicious, _write_summary


# ---------------------------------------------------------------------------
# _eval_split
# ---------------------------------------------------------------------------

def _make_test_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "text": [f"text number {i} is here" for i in range(n)],
        "label": rng.integers(0, 2, n).tolist(),
        "language": ["en"] * n,
        "source_dataset": ["multitude"] * n,
        "domain": ["news"] * n,
        "generator": rng.choice(["human", "gpt4", "llama"], n).tolist(),
        "pred_label": rng.integers(0, 2, n).tolist(),
        "proba_machine": rng.random(n).tolist(),
    })


class TestEvalSplit:
    def test_output_schema(self):
        df = _make_test_df()
        result = _eval_split(df, "pred_label", "proba_machine", "test_en", "supervised")
        assert result["split"] == "test_en"
        assert result["model"] == "supervised"
        assert "overall" in result
        assert "per_generator" in result
        overall = result["overall"]
        assert "n" in overall
        assert "accuracy" in overall
        assert "f1" in overall
        assert "auroc" in overall

    def test_per_generator_keys_match_data(self):
        df = _make_test_df()
        result = _eval_split(df, "pred_label", "proba_machine", "test_en", "supervised")
        found = set(result["per_generator"].keys())
        expected = set(df["generator"].unique())
        assert found == expected

    def test_deterministic(self):
        df = _make_test_df()
        r1 = _eval_split(df, "pred_label", "proba_machine", "test_en", "supervised")
        r2 = _eval_split(df, "pred_label", "proba_machine", "test_en", "supervised")
        assert r1 == r2

    def test_overall_n_matches_df_length(self):
        df = _make_test_df(30)
        result = _eval_split(df, "pred_label", "proba_machine", "test_en", "supervised")
        assert result["overall"]["n"] == 30


# ---------------------------------------------------------------------------
# _flag_suspicious
# ---------------------------------------------------------------------------

class TestFlagSuspicious:
    def test_near_chance_flagged(self):
        metrics = {"overall": {"accuracy": 0.51}}
        warnings = _flag_suspicious(metrics, "test_de", "supervised")
        assert len(warnings) == 1
        assert "near chance" in warnings[0].lower() or "WARNING" in warnings[0]

    def test_near_perfect_flagged(self):
        metrics = {"overall": {"accuracy": 0.999}}
        warnings = _flag_suspicious(metrics, "test_de", "supervised")
        assert len(warnings) == 1
        assert "near-perfect" in warnings[0].lower() or "WARNING" in warnings[0]

    def test_normal_accuracy_not_flagged(self):
        metrics = {"overall": {"accuracy": 0.80}}
        warnings = _flag_suspicious(metrics, "test_en", "supervised")
        assert warnings == []

    def test_missing_accuracy_no_crash(self):
        metrics = {"overall": {}}
        warnings = _flag_suspicious(metrics, "test_en", "supervised")
        assert warnings == []


# ---------------------------------------------------------------------------
# _write_summary
# ---------------------------------------------------------------------------

class TestWriteSummary:
    def _make_report(self) -> dict:
        df = _make_test_df()
        r = _eval_split(df, "pred_label", "proba_machine", "test_en", "supervised")
        return {
            "generated_at": "2026-01-01T00:00:00Z",
            "results": [r],
            "warnings": [],
        }

    def test_writes_file(self):
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.txt"
            _write_summary(report, path)
            assert path.exists()
            content = path.read_text()
            assert "supervised" in content
            assert "test_en" in content

    def test_warnings_appear_in_output(self):
        report = self._make_report()
        report["warnings"] = ["WARNING: something suspicious"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.txt"
            _write_summary(report, path)
            content = path.read_text()
            assert "something suspicious" in content
