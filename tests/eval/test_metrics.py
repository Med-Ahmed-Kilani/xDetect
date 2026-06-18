"""Unit tests for the eval metrics module."""
import numpy as np
import pytest

from src.eval.metrics import compute_metrics, compute_per_generator


class TestComputeMetrics:
    def test_perfect_predictions(self):
        labels = np.array([0, 1, 0, 1])
        preds  = np.array([0, 1, 0, 1])
        proba  = np.array([0.1, 0.9, 0.1, 0.9])
        m = compute_metrics(labels, preds, proba)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["auroc"] == pytest.approx(1.0)

    def test_n_matches_input_length(self):
        labels = np.array([0, 0, 1, 1, 0])
        preds  = np.array([0, 1, 1, 0, 0])
        m = compute_metrics(labels, preds)
        assert m["n"] == 5

    def test_auroc_none_when_proba_not_given(self):
        labels = np.array([0, 1])
        preds  = np.array([0, 1])
        m = compute_metrics(labels, preds)
        assert m["auroc"] is None

    def test_auroc_none_when_single_class(self):
        labels = np.array([1, 1, 1])
        preds  = np.array([1, 1, 0])
        proba  = np.array([0.9, 0.8, 0.3])
        m = compute_metrics(labels, preds, proba)
        assert m["auroc"] is None

    def test_all_wrong_predictions(self):
        labels = np.array([0, 0, 1, 1])
        preds  = np.array([1, 1, 0, 0])
        m = compute_metrics(labels, preds)
        assert m["accuracy"] == pytest.approx(0.0)


class TestComputePerGenerator:
    def test_splits_by_generator(self):
        labels     = np.array([0, 1, 0, 1])
        preds      = np.array([0, 1, 1, 1])
        generators = np.array(["human", "gpt4", "human", "gpt4"])
        result = compute_per_generator(labels, preds, generators)
        assert "human" in result
        assert "gpt4" in result
        assert result["human"]["n"] == 2
        assert result["gpt4"]["n"] == 2

    def test_per_generator_accuracy(self):
        labels     = np.array([0, 0, 1, 1])
        preds      = np.array([0, 0, 1, 0])  # gpt4 has 1 wrong
        generators = np.array(["human", "human", "gpt4", "gpt4"])
        result = compute_per_generator(labels, preds, generators)
        assert result["human"]["accuracy"] == pytest.approx(1.0)
        assert result["gpt4"]["accuracy"] == pytest.approx(0.5)
