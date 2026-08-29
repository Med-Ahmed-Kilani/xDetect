"""
Integration test for the Month 3 scalability runner (Step 6), on tiny
synthetic config — verifies wiring (zero-shot / few-shot-adapter /
full-retrain conditions, --lang filtering + merge) without any GPU work.
AdapterTrainer, FullRetrainer, and evaluate_checkpoint are all mocked.
"""
import contextlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.adapters import scalability_runner as sr

_SEEDS = [42, 123]
_ADAPTERS_CFG = {
    "scalability": {
        "target_languages": ["de", "ar"],
        "data_sizes": [100, 500, "full"],
        "backbone_checkpoint": "data/checkpoints/xlmr_base",
        "backbone_key": "xlmr_base",
        "adapter_checkpoint_template": "data/checkpoints/adapters/{lang}_{n}",
        "full_retrain_checkpoint_template": "data/checkpoints/full_retrain/{lang}",
        "report_path": "reports/scalability_results.json",
        "seeds": _SEEDS,
    }
}
_DATASETS_CFG = {
    "processed": {
        "train_template": "data/processed/train_{lang}.parquet",
        "test_template": "data/processed/test_{lang}.parquet",
    }
}


def _fake_metrics(tag: str) -> dict:
    return {"overall": {"accuracy": 0.9, "f1": 0.9, "auroc": 0.9, "n": 10, "tag": tag},
            "per_generator": {}}


@contextlib.contextmanager
def _runner_ctx(tmp_path):
    def fake_load_config(name):
        return _ADAPTERS_CFG if name == "adapters" else _DATASETS_CFG

    def fake_resolve_path(relative):
        return tmp_path / str(relative)

    trainer_mock = MagicMock()
    trainer_mock.train.return_value = None
    retrainer_mock = MagicMock()
    retrainer_mock.train.return_value = None

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("src.adapters.scalability_runner.load_config", side_effect=fake_load_config))
        stack.enter_context(patch("src.adapters.scalability_runner.resolve_path", side_effect=fake_resolve_path))
        stack.enter_context(patch("src.adapters.scalability_runner.AdapterTrainer", return_value=trainer_mock))
        stack.enter_context(patch("src.adapters.scalability_runner.FullRetrainer", return_value=retrainer_mock))
        p_eval = stack.enter_context(patch(
            "src.adapters.scalability_runner.evaluate_checkpoint",
            side_effect=lambda checkpoint_path, **kw: _fake_metrics(str(checkpoint_path)),
        ))
        yield {"eval": p_eval, "trainer": trainer_mock, "retrainer": retrainer_mock}


class TestRunZeroShot:
    def test_evaluates_backbone_per_language(self, tmp_path):
        with _runner_ctx(tmp_path):
            results = sr.run_zero_shot(["de", "ar"])
        assert set(results.keys()) == {"de", "ar"}
        for lang_result in results.values():
            # aggregated across seeds: mean ± std + per-seed breakdown
            agg = lang_result["overall"]["accuracy"]
            assert agg["mean"] == 0.9
            assert agg["std"] == 0.0
            assert lang_result["overall"]["seeds"] == _SEEDS
            assert set(lang_result["per_seed"].keys()) == {str(s) for s in _SEEDS}


class TestRunFewShotAdapters:
    def test_all_lang_x_size_combinations_present(self, tmp_path):
        with _runner_ctx(tmp_path) as mocks:
            results = sr.run_few_shot_adapters(["de", "ar"], [100, 500, "full"])
        assert set(results.keys()) == {"de", "ar"}
        for lang in ["de", "ar"]:
            assert set(results[lang].keys()) == {"100", "500", "full"}
            for agg in results[lang].values():
                assert agg["overall"]["seeds"] == _SEEDS
        # one train() call per (language, size, seed) triple
        assert mocks["trainer"].train.call_count == 2 * 3 * len(_SEEDS)


class TestRunFullRetrain:
    def test_one_result_per_language(self, tmp_path):
        with _runner_ctx(tmp_path) as mocks:
            results = sr.run_full_retrain(["de", "ar"])
        assert set(results.keys()) == {"de", "ar"}
        for agg in results.values():
            assert agg["overall"]["seeds"] == _SEEDS
        assert mocks["retrainer"].train.call_count == 2 * len(_SEEDS)


class TestRunFull:
    def test_writes_valid_json_with_all_12_conditions(self, tmp_path):
        with _runner_ctx(tmp_path):
            report_path = sr.run()

        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert set(data.keys()) == {"zero_shot", "few_shot_adapter", "full_retrain"}
        assert set(data["zero_shot"].keys()) == {"de", "ar"}
        assert sum(len(v) for v in data["few_shot_adapter"].values()) == 6
        assert set(data["full_retrain"].keys()) == {"de", "ar"}

    def test_deterministic_across_runs(self, tmp_path):
        with _runner_ctx(tmp_path):
            sr.run()
            first = (tmp_path / "reports/scalability_results.json").read_text()
            sr.run()
            second = (tmp_path / "reports/scalability_results.json").read_text()
        assert first == second

    def test_invalid_lang_raises(self, tmp_path):
        with _runner_ctx(tmp_path):
            with pytest.raises(ValueError, match="not a valid target language"):
                sr.run(lang="fr")

    def test_lang_filter_merges_with_existing_report(self, tmp_path):
        report_path = tmp_path / "reports/scalability_results.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps({
            "zero_shot": {"ar": {"overall": {"accuracy": 0.5}}},
            "few_shot_adapter": {"ar": {"100": {"overall": {"accuracy": 0.5}}}},
            "full_retrain": {"ar": {"overall": {"accuracy": 0.5}}},
        }))

        with _runner_ctx(tmp_path):
            sr.run(lang="de")

        data = json.loads(report_path.read_text())
        # "ar" results from the prior session must survive untouched
        assert data["zero_shot"]["ar"]["overall"]["accuracy"] == 0.5
        # "de" results from this session must now be present
        assert "de" in data["zero_shot"]
        assert "de" in data["full_retrain"]
