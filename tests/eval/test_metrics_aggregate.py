"""
Tests for the multi-seed aggregation helpers in src/eval/metrics.py:
  - aggregate_seed_metrics: mean / sample-std / value list, None handling
  - mean_of: normalises scalar and {"mean": ..} entries
  - fmt_mean_std: "mean ± std" rendering, scalar + legacy fallbacks
"""
import math

from src.eval.metrics import aggregate_seed_metrics, fmt_mean_std, mean_of


class TestAggregateSeedMetrics:
    def test_mean_and_sample_std(self):
        per_seed = {
            42:  {"accuracy": 0.90, "f1": 0.80, "auroc": 0.95, "n": 100},
            123: {"accuracy": 0.92, "f1": 0.82, "auroc": 0.96, "n": 100},
            456: {"accuracy": 0.94, "f1": 0.84, "auroc": 0.97, "n": 100},
        }
        agg = aggregate_seed_metrics(per_seed)
        assert agg["seeds"] == [42, 123, 456]
        assert math.isclose(agg["accuracy"]["mean"], 0.92)
        # sample std (ddof=1) of [0.90, 0.92, 0.94] == 0.02
        assert math.isclose(agg["accuracy"]["std"], 0.02, abs_tol=1e-9)
        assert agg["accuracy"]["values"] == [0.90, 0.92, 0.94]
        assert agg["n"] == 100

    def test_single_seed_std_is_zero(self):
        agg = aggregate_seed_metrics({42: {"accuracy": 0.9, "f1": 0.9, "auroc": 0.9, "n": 10}})
        assert agg["accuracy"]["mean"] == 0.9
        assert agg["accuracy"]["std"] == 0.0
        assert agg["seeds"] == [42]

    def test_none_metric_values_dropped(self):
        per_seed = {
            42:  {"accuracy": 0.9, "f1": 0.8, "auroc": None, "n": 5},
            123: {"accuracy": 0.8, "f1": 0.7, "auroc": 0.9, "n": 5},
        }
        agg = aggregate_seed_metrics(per_seed)
        # only one seed had an AUROC → mean is that value, std 0.0
        assert agg["auroc"]["values"] == [0.9]
        assert agg["auroc"]["mean"] == 0.9
        assert agg["auroc"]["std"] == 0.0
        # accuracy had both seeds
        assert agg["accuracy"]["values"] == [0.9, 0.8]

    def test_all_none_metric_gives_none_mean(self):
        per_seed = {
            42:  {"accuracy": 0.9, "f1": 0.8, "auroc": None, "n": 5},
            123: {"accuracy": 0.8, "f1": 0.7, "auroc": None, "n": 5},
        }
        agg = aggregate_seed_metrics(per_seed)
        assert agg["auroc"]["mean"] is None
        assert agg["auroc"]["std"] is None


class TestMeanOf:
    def test_scalar_passthrough(self):
        assert mean_of(0.42) == 0.42

    def test_dict_returns_mean(self):
        assert mean_of({"mean": 0.5, "std": 0.1}) == 0.5

    def test_none_is_none(self):
        assert mean_of(None) is None


class TestFmtMeanStd:
    def test_dict_renders_mean_and_std(self):
        assert fmt_mean_std({"mean": 0.8, "std": 0.03}) == "0.8000 ± 0.0300"

    def test_scalar_renders_bare(self):
        assert fmt_mean_std(0.8) == "0.8000"

    def test_none_is_na(self):
        assert fmt_mean_std(None) == "N/A"

    def test_dict_missing_mean_is_na(self):
        assert fmt_mean_std({"mean": None, "std": None}) == "N/A"
