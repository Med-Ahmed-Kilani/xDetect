"""
Tests for FullRetrainer (src/adapters/full_retrain.py):
  - warm-starts from the backbone checkpoint path (not an HF model id)
  - trainable_parameters() always reports fraction 1.0 (full fine-tune)
"""
from unittest.mock import MagicMock, patch

import torch

from src.adapters.full_retrain import FullRetrainer

_CFG = {
    "num_epochs": 1, "learning_rate": 2e-5, "batch_size": 16,
    "warmup_ratio": 0.1, "weight_decay": 0.01, "adam_epsilon": 1e-8, "seed": 42,
}


class TestWarmStart:
    def test_model_id_is_backbone_checkpoint_path(self, tmp_path):
        backbone = tmp_path / "xlmr_base"
        retrainer = FullRetrainer(
            backbone_path=backbone, lang="de", output_dir=tmp_path / "out", cfg=_CFG,
        )
        assert retrainer.baseline.model_id == str(backbone)

    def test_checkpoint_dir_is_per_language_output_dir(self, tmp_path):
        out_dir = tmp_path / "full_retrain_de"
        with patch("src.baselines.supervised_baseline.resolve_path", side_effect=lambda p: p):
            retrainer = FullRetrainer(
                backbone_path=tmp_path / "xlmr_base", lang="de", output_dir=out_dir, cfg=_CFG,
            )
        assert str(retrainer.baseline.checkpoint_dir) == str(out_dir)


class TestTrainableParameters:
    def test_reports_full_fraction(self, tmp_path):
        retrainer = FullRetrainer(
            backbone_path=tmp_path / "xlmr_base", lang="de", output_dir=tmp_path / "out", cfg=_CFG,
        )
        fake_model = MagicMock()
        fake_model.parameters.return_value = [torch.zeros(10), torch.zeros(5)]
        retrainer.baseline._model = fake_model

        total, fraction = retrainer.trainable_parameters()
        assert total == 15
        assert fraction == 1.0
