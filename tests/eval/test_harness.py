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

import torch

from src.eval.harness import _eval_split, _flag_suspicious, _write_summary, evaluate_checkpoint


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
        metrics = {"overall": {"accuracy": 0.80, "auroc": 0.85}}
        warnings = _flag_suspicious(metrics, "test_en", "supervised")
        assert warnings == []

    def test_missing_accuracy_no_crash(self):
        metrics = {"overall": {}}
        warnings = _flag_suspicious(metrics, "test_en", "supervised")
        assert warnings == []

    def test_divergence_high_acc_low_auroc_flagged(self):
        metrics = {"overall": {"accuracy": 0.85, "auroc": 0.70}}
        warnings = _flag_suspicious(metrics, "test_de", "supervised")
        assert len(warnings) == 1
        assert "confusion matrix" in warnings[0]

    def test_divergence_not_flagged_when_auroc_acceptable(self):
        metrics = {"overall": {"accuracy": 0.85, "auroc": 0.80}}
        warnings = _flag_suspicious(metrics, "test_de", "supervised")
        assert warnings == []

    def test_divergence_not_flagged_when_accuracy_below_threshold(self):
        metrics = {"overall": {"accuracy": 0.75, "auroc": 0.60}}
        warnings = _flag_suspicious(metrics, "test_de", "supervised")
        assert warnings == []

    def test_divergence_not_flagged_when_auroc_missing(self):
        metrics = {"overall": {"accuracy": 0.90, "auroc": None}}
        warnings = _flag_suspicious(metrics, "test_de", "supervised")
        assert warnings == []


# ---------------------------------------------------------------------------
# _write_summary
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# evaluate_checkpoint (Month 3 adapter/full-checkpoint eval extension)
# ---------------------------------------------------------------------------

def _proba_to_logits(proba: list[float]) -> torch.Tensor:
    """Build 2-class logits whose softmax reproduces `proba` for class 1 exactly."""
    p = torch.tensor(proba, dtype=torch.float32)
    logit_1 = torch.log(p / (1 - p))
    logit_0 = torch.zeros_like(logit_1)
    return torch.stack([logit_0, logit_1], dim=1)


def _make_eval_test_parquet(tmp_path) -> tuple[str, np.ndarray, np.ndarray]:
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    generators = ["human", "gpt", "human", "gpt", "human", "gpt", "human", "gpt"]
    df = pd.DataFrame({
        "text": [f"sample text {i}" for i in range(8)],
        "label": labels,
        "generator": generators,
    })
    path = tmp_path / "test_de.parquet"
    df.to_parquet(path, index=False)
    return str(path), np.array(labels), np.array(generators)


def _make_harness_tok_mock():
    tok = MagicMock()
    tok.side_effect = lambda texts, **kw: {
        "input_ids": torch.zeros(len(texts), 8, dtype=torch.long),
        "attention_mask": torch.ones(len(texts), 8, dtype=torch.long),
    }
    return tok


class TestEvaluateCheckpointFull:
    def test_matches_manually_computed_metrics(self, tmp_path):
        """
        Regression check for the harness extension: given fixed logits, the
        metrics evaluate_checkpoint returns must equal sklearn computed
        directly on the same proba/preds/labels — proving the new code path
        doesn't corrupt the existing metric computation.
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        test_path, labels, generators = _make_eval_test_parquet(tmp_path)
        proba = [0.1, 0.9, 0.2, 0.8, 0.6, 0.4, 0.3, 0.7]

        model_mock = MagicMock()
        model_mock.side_effect = lambda **batch: MagicMock(logits=_proba_to_logits(proba))
        model_mock.to = MagicMock(return_value=model_mock)
        model_mock.eval = MagicMock()

        tok_mock = _make_harness_tok_mock()

        with patch("src.eval.harness.AutoModelForSequenceClassification") as p_model, \
             patch("src.eval.harness.AutoTokenizer") as p_tok:
            p_model.from_pretrained.return_value = model_mock
            p_tok.from_pretrained.return_value = tok_mock

            result = evaluate_checkpoint(
                checkpoint_path="fake/checkpoint", test_parquet=test_path, is_adapter=False,
            )

        preds = (np.array(proba) >= 0.5).astype(int)
        assert result["overall"]["accuracy"] == pytest.approx(accuracy_score(labels, preds))
        assert result["overall"]["f1"] == pytest.approx(f1_score(labels, preds))
        assert result["overall"]["auroc"] == pytest.approx(roc_auc_score(labels, proba))
        assert set(result["per_generator"].keys()) == set(generators.tolist())

    def test_loads_checkpoint_directly_without_backbone(self, tmp_path):
        test_path, _, _ = _make_eval_test_parquet(tmp_path)
        model_mock = MagicMock()
        model_mock.side_effect = lambda **batch: MagicMock(
            logits=_proba_to_logits([0.5] * 8)
        )
        model_mock.to = MagicMock(return_value=model_mock)

        with patch("src.eval.harness.AutoModelForSequenceClassification") as p_model, \
             patch("src.eval.harness.AutoTokenizer") as p_tok:
            p_model.from_pretrained.return_value = model_mock
            p_tok.from_pretrained.return_value = _make_harness_tok_mock()

            evaluate_checkpoint(checkpoint_path="some/ckpt", test_parquet=test_path)

            p_model.from_pretrained.assert_called_once_with("some/ckpt", num_labels=2)


class TestEvaluateCheckpointAdapter:
    def test_requires_backbone_path(self, tmp_path):
        test_path, _, _ = _make_eval_test_parquet(tmp_path)
        with pytest.raises(ValueError, match="backbone_path"):
            evaluate_checkpoint(
                checkpoint_path="adapter/ckpt", test_parquet=test_path, is_adapter=True,
            )

    def test_loads_via_lora_adapter(self, tmp_path):
        test_path, _, _ = _make_eval_test_parquet(tmp_path)
        model_mock = MagicMock()
        model_mock.side_effect = lambda **batch: MagicMock(
            logits=_proba_to_logits([0.5] * 8)
        )
        model_mock.to = MagicMock(return_value=model_mock)

        adapter_mock = MagicMock()
        adapter_mock.model = model_mock
        adapter_mock.tokenizer = _make_harness_tok_mock()

        with patch("src.adapters.lora_adapter.LoRAAdapter", return_value=adapter_mock) as p_cls:
            result = evaluate_checkpoint(
                checkpoint_path="adapter/ckpt", test_parquet=test_path,
                is_adapter=True, backbone_path="backbone/ckpt",
            )

        p_cls.assert_called_once_with("backbone/ckpt", num_labels=2)
        adapter_mock.load.assert_called_once_with("adapter/ckpt")
        assert "overall" in result


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
