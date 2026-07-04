"""
Full-retrain condition for the scalability experiment: fine-tunes the
*entire* backbone (all parameters unfrozen) on the full target-language
training set, using the backbone checkpoint as a warm start rather than
downloading the HF pretrained weights again.

Implemented as a thin wrapper around SupervisedBaseline — full fine-tuning
is exactly what SupervisedBaseline already does; the only difference here
is that model_id points at a local checkpoint dir (warm start) instead of
an HF model id, and hyperparameters come from configs/adapters.yaml's
full_retrain section.
"""
import logging
from pathlib import Path
from typing import Optional

from src.baselines.supervised_baseline import SupervisedBaseline
from src.config import load_config

logger = logging.getLogger(__name__)


class FullRetrainer:
    def __init__(self, backbone_path: str | Path, lang: str, output_dir: str | Path,
                 cfg: Optional[dict] = None, num_labels: int = 2, max_length: int = 512):
        if cfg is None:
            cfg = load_config("adapters")["full_retrain"]
        self.backbone_path = Path(backbone_path)
        self.lang = lang
        self.output_dir = Path(output_dir)
        self.cfg = cfg

        sup_cfg = {
            "model_id": str(self.backbone_path),
            "num_labels": num_labels,
            "max_length": max_length,
            "batch_size": cfg["batch_size"],
            "learning_rate": cfg["learning_rate"],
            "num_epochs": cfg["num_epochs"],
            "warmup_ratio": cfg["warmup_ratio"],
            "weight_decay": cfg["weight_decay"],
            "adam_epsilon": cfg.get("adam_epsilon", 1e-8),
            "seed": cfg["seed"],
            "fp16": False,
            "checkpoint_dir": str(self.output_dir),
        }
        self.baseline = SupervisedBaseline(cfg=sup_cfg)

    def train(self, train_parquet: str | Path) -> Path:
        logger.info("Full-retrain: warm-starting from %s for lang=%s …",
                    self.backbone_path, self.lang)
        return self.baseline.train(train_parquet)

    def evaluate(self, test_parquet: str | Path) -> dict:
        return self.baseline.evaluate(test_parquet)

    def load(self, checkpoint_dir: Optional[str | Path] = None) -> None:
        self.baseline.load(checkpoint_dir or self.output_dir)

    def trainable_parameters(self) -> tuple[int, float]:
        """Full retrain unfreezes every parameter — always returns fraction 1.0."""
        if self.baseline._model is None:
            raise RuntimeError("Model not loaded. Call .train() or .load() first.")
        total = sum(p.numel() for p in self.baseline._model.parameters())
        return total, 1.0
