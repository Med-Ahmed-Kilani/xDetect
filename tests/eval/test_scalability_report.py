"""
Tests for src/eval/scalability_report.py:
  - compute_data_efficiency_curve orders sizes correctly ("full" last)
  - compute_crossover finds the smallest n that beats zero-shot
  - declare_winner picks the max AUROC condition, with a tie threshold
  - generate() end-to-end produces a report whose numbers match the JSON source
"""
import json

import pytest

from src.eval.metrics import aggregate_seed_metrics
from src.eval.scalability_report import (
    compute_crossover,
    compute_data_efficiency_curve,
    declare_winner,
    generate,
)


def _m(auroc):
    return {"overall": {"accuracy": 0.8, "f1": 0.8, "auroc": auroc}}


class TestDataEfficiencyCurve:
    def test_orders_full_last(self):
        few_shot_lang = {
            "500": _m(0.85), "full": _m(0.95), "100": _m(0.70), "1000": _m(0.90),
        }
        curve = compute_data_efficiency_curve(few_shot_lang)
        assert [n for n, _ in curve] == ["100", "500", "1000", "full"]

    def test_values_match_source(self):
        few_shot_lang = {"100": _m(0.70), "full": _m(0.95)}
        curve = compute_data_efficiency_curve(few_shot_lang)
        assert dict(curve) == {"100": 0.70, "full": 0.95}


class TestCrossover:
    def test_finds_smallest_n_exceeding_zero_shot(self):
        curve = [("100", 0.60), ("500", 0.75), ("1000", 0.80), ("full", 0.90)]
        assert compute_crossover(curve, zero_shot_value=0.70) == "500"

    def test_returns_none_if_never_exceeds(self):
        curve = [("100", 0.60), ("500", 0.65)]
        assert compute_crossover(curve, zero_shot_value=0.90) is None

    def test_returns_none_if_zero_shot_missing(self):
        curve = [("100", 0.60)]
        assert compute_crossover(curve, zero_shot_value=None) is None


class TestDeclareWinner:
    def test_picks_highest_auroc(self):
        winner, scores = declare_winner(zero_shot_auroc=0.70, full_retrain_auroc=0.95,
                                        best_adapter_auroc=0.85)
        assert winner == "full_retrain"
        assert scores["full_retrain"] == 0.95

    def test_declares_tie_within_threshold(self):
        winner, _ = declare_winner(zero_shot_auroc=0.70, full_retrain_auroc=0.900,
                                   best_adapter_auroc=0.899)
        assert "tie" in winner

    def test_handles_missing_values(self):
        winner, scores = declare_winner(zero_shot_auroc=None, full_retrain_auroc=None,
                                        best_adapter_auroc=0.85)
        assert winner == "few_shot_adapter"
        assert scores == {"few_shot_adapter": 0.85}

    def test_all_missing_is_inconclusive(self):
        winner, scores = declare_winner(None, None, None)
        assert "inconclusive" in winner
        assert scores == {}


class TestGenerateEndToEnd:
    def _write_results(self, tmp_path) -> str:
        results = {
            "zero_shot": {
                "de": _m(0.75),
                "ar": _m(0.60),
            },
            "few_shot_adapter": {
                "de": {"100": _m(0.65), "500": _m(0.80), "full": _m(0.92)},
                "ar": {"100": _m(0.55), "500": _m(0.58), "full": _m(0.62)},
            },
            "full_retrain": {
                "de": _m(0.94),
                "ar": _m(0.90),
            },
        }
        path = tmp_path / "scalability_results.json"
        path.write_text(json.dumps(results))
        return path

    def test_report_generated_and_numbers_match_source(self, tmp_path):
        results_path = self._write_results(tmp_path)
        out_path = tmp_path / "scalability_report.txt"

        report_path = generate(results_json=results_path, out=out_path)
        assert report_path == out_path
        assert report_path.exists()

        content = report_path.read_text()
        assert "de" in content and "ar" in content
        assert "0.7500" in content   # de zero-shot AUROC
        assert "0.9200" in content   # de full-data adapter AUROC
        assert "0.9400" in content   # de full-retrain AUROC
        assert "Crossover" in content
        assert "WINNER DECLARATION" in content

    def test_de_crossover_at_n_500(self, tmp_path):
        results_path = self._write_results(tmp_path)
        out_path = tmp_path / "scalability_report.txt"
        generate(results_json=results_path, out=out_path)
        content = out_path.read_text()
        # de zero-shot AUROC=0.75; adapter at n=100 is 0.65 (below), n=500 is 0.80 (above)
        assert "at n=500" in content


class TestGenerateMultiSeed:
    """generate() renders mean ± std for every condition and still computes
    curves / crossover / winner from the across-seed means."""

    def _agg(self, u):
        return {"overall": aggregate_seed_metrics({
            42:  {"accuracy": u,        "f1": u,        "auroc": u},
            123: {"accuracy": u - 0.02, "f1": u - 0.02, "auroc": u - 0.02},
            456: {"accuracy": u + 0.02, "f1": u + 0.02, "auroc": u + 0.02},
        })}

    def _write(self, tmp_path):
        results = {
            "zero_shot": {"de": self._agg(0.75), "ar": self._agg(0.60)},
            "few_shot_adapter": {
                "de": {"100": self._agg(0.65), "500": self._agg(0.80), "full": self._agg(0.92)},
                "ar": {"100": self._agg(0.55), "500": self._agg(0.58), "full": self._agg(0.62)},
            },
            "full_retrain": {"de": self._agg(0.94), "ar": self._agg(0.90)},
        }
        path = tmp_path / "scalability_results.json"
        path.write_text(json.dumps(results))
        return path

    def test_mean_std_rendered_and_logic_uses_means(self, tmp_path):
        out_path = tmp_path / "report.txt"
        generate(results_json=self._write(tmp_path), out=out_path)
        content = out_path.read_text()

        assert "Seeds per condition: 3" in content
        assert "0.7500 ± 0.0200" in content          # de zero-shot mean ± std
        assert "0.9200 ± 0.0200" in content          # de full-data adapter
        # crossover still derived from across-seed means
        assert "at n=500" in content
        assert "de: full_retrain" in content
