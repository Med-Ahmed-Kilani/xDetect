"""
Tests for src/eval/backbone_report.py — winner selection logic.

All tests mock the results dict; no real model checkpoints or JSON files needed.
"""
import pytest

from src.eval.backbone_report import select_winner, CLOSE_THRESHOLD


# ---------------------------------------------------------------------------
# Clear winner
# ---------------------------------------------------------------------------

class TestSelectWinnerClearWinner:
    def test_xlmr_wins_by_f1(self):
        results = {
            "mbert":          {"en": {"f1": 0.70}, "de": {"f1": 0.65}, "ar": {"f1": 0.60}},
            "xlmr_base":      {"en": {"f1": 0.85}, "de": {"f1": 0.82}, "ar": {"f1": 0.80}},
            "mdeberta_v3_base": {"en": {"f1": 0.75}, "de": {"f1": 0.70}, "ar": {"f1": 0.68}},
        }
        winner, means, is_tie = select_winner(results, "f1")
        assert winner == "xlmr_base"
        assert not is_tie
        assert means["xlmr_base"] > means["mbert"]
        assert means["xlmr_base"] > means["mdeberta_v3_base"]

    def test_mdeberta_wins_by_f1(self):
        results = {
            "mbert":          {"en": {"f1": 0.70}, "de": {"f1": 0.65}, "ar": {"f1": 0.60}},
            "xlmr_base":      {"en": {"f1": 0.75}, "de": {"f1": 0.72}, "ar": {"f1": 0.70}},
            "mdeberta_v3_base": {"en": {"f1": 0.90}, "de": {"f1": 0.88}, "ar": {"f1": 0.85}},
        }
        winner, means, is_tie = select_winner(results, "f1")
        assert winner == "mdeberta_v3_base"
        assert not is_tie

    def test_mbert_wins_by_f1(self):
        results = {
            "mbert":          {"en": {"f1": 0.92}, "de": {"f1": 0.90}, "ar": {"f1": 0.88}},
            "xlmr_base":      {"en": {"f1": 0.80}, "de": {"f1": 0.78}, "ar": {"f1": 0.75}},
            "mdeberta_v3_base": {"en": {"f1": 0.70}, "de": {"f1": 0.68}, "ar": {"f1": 0.65}},
        }
        winner, means, is_tie = select_winner(results, "f1")
        assert winner == "mbert"
        assert not is_tie

    def test_mean_is_correct(self):
        results = {
            "mbert":     {"en": {"f1": 0.60}, "de": {"f1": 0.80}, "ar": {"f1": 0.70}},
            "xlmr_base": {"en": {"f1": 0.50}, "de": {"f1": 0.50}, "ar": {"f1": 0.50}},
        }
        _, means, _ = select_winner(results, "f1")
        assert abs(means["mbert"]     - (0.60 + 0.80 + 0.70) / 3) < 1e-9
        assert abs(means["xlmr_base"] - 0.50) < 1e-9


# ---------------------------------------------------------------------------
# Tie detection
# ---------------------------------------------------------------------------

class TestSelectWinnerTie:
    def test_tie_when_within_threshold(self):
        delta = CLOSE_THRESHOLD / 2   # half the threshold → tie
        results = {
            "mbert":     {"en": {"f1": 0.80},        "de": {"f1": 0.80},        "ar": {"f1": 0.80}},
            "xlmr_base": {"en": {"f1": 0.80 + delta}, "de": {"f1": 0.80 + delta}, "ar": {"f1": 0.80 + delta}},
        }
        winner, means, is_tie = select_winner(results, "f1")
        assert is_tie

    def test_no_tie_when_just_above_threshold(self):
        delta = CLOSE_THRESHOLD + 0.001   # just above threshold → clear winner
        results = {
            "mbert":     {"en": {"f1": 0.80},        "de": {"f1": 0.80},        "ar": {"f1": 0.80}},
            "xlmr_base": {"en": {"f1": 0.80 + delta}, "de": {"f1": 0.80 + delta}, "ar": {"f1": 0.80 + delta}},
        }
        _, _, is_tie = select_winner(results, "f1")
        assert not is_tie

    def test_exact_tie_declared_as_tie(self):
        # Perfectly equal means → is_tie (difference is 0 < CLOSE_THRESHOLD)
        results = {
            "mbert":     {"en": {"f1": 0.75}, "de": {"f1": 0.75}, "ar": {"f1": 0.75}},
            "xlmr_base": {"en": {"f1": 0.75}, "de": {"f1": 0.75}, "ar": {"f1": 0.75}},
        }
        _, _, is_tie = select_winner(results, "f1")
        assert is_tie


# ---------------------------------------------------------------------------
# Robustness: missing metric values
# ---------------------------------------------------------------------------

class TestSelectWinnerMissingValues:
    def test_none_values_skipped(self):
        results = {
            "mbert":     {"en": {"f1": 0.80}, "de": {"f1": None},  "ar": {"f1": 0.80}},
            "xlmr_base": {"en": {"f1": 0.70}, "de": {"f1": 0.70},  "ar": {"f1": 0.70}},
        }
        winner, means, _ = select_winner(results, "f1")
        # mBERT mean is (0.80 + 0.80) / 2 = 0.80; XLM-R is 0.70
        assert winner == "mbert"
        assert abs(means["mbert"] - 0.80) < 1e-9

    def test_completely_missing_language_returns_zero(self):
        results = {
            "mbert":     {},
            "xlmr_base": {"en": {"f1": 0.80}},
        }
        winner, means, _ = select_winner(results, "f1")
        assert winner == "xlmr_base"
        assert means["mbert"] == 0.0


# ---------------------------------------------------------------------------
# AUROC as ranking metric
# ---------------------------------------------------------------------------

class TestSelectWinnerByAUROC:
    def test_winner_selected_by_auroc(self):
        results = {
            "mbert":     {"en": {"auroc": 0.60}, "de": {"auroc": 0.65}, "ar": {"auroc": 0.62}},
            "xlmr_base": {"en": {"auroc": 0.90}, "de": {"auroc": 0.92}, "ar": {"auroc": 0.88}},
        }
        winner, _, is_tie = select_winner(results, "auroc")
        assert winner == "xlmr_base"
        assert not is_tie
