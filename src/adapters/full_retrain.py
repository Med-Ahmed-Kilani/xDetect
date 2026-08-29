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
                 cfg: Optional[dict] = None, num_labels: int = 2, max_length: int = 512,
                 seed: Optional[int] = None):
        if cfg is None:
            cfg = load_config("adapters")["full_retrain"]
        self.backbone_path = Path(backbone_path)
        self.lang = lang
        self.output_dir = Path(output_dir)
        self.cfg = cfg

        # seeds: multi-seed runs fine-tune once per seed. A legacy single "seed"
        # key is accepted as a one-element list; the `seed` ctor arg pins the
        # active seed for this instance (used by scalability_runner).
        seeds = list(cfg["seeds"]) if "seeds" in cfg else [cfg["seed"]]

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
            "seeds": seeds,
            "fp16": False,
            "checkpoint_dir": str(self.output_dir),
        }
        self.baseline = SupervisedBaseline(cfg=sup_cfg)
        self.seeds = self.baseline.seeds
        if seed is not None:
            self.baseline.seed = seed
        self.seed = self.baseline.seed

    def seed_output_dir(self, seed: int) -> Path:
        return self.baseline.seed_checkpoint_dir(seed)

    def is_seed_complete(self, seed: int) -> bool:
        return self.baseline.is_seed_complete(seed)

    def train(self, train_parquet: str | Path, *,
              seed: int | None = None,
              output_dir: str | Path | None = None) -> Path:
        logger.info("Full-retrain: warm-starting from %s for lang=%s (seed=%s) …",
                    self.backbone_path, self.lang, seed if seed is not None else self.seed)
        return self.baseline.train(
            train_parquet,
            seed=self.seed if seed is None else seed,
            checkpoint_dir=output_dir,
        )

    def train_seeds(self, train_parquet: str | Path) -> dict[int, Path]:
        """
        Fine-tune the whole model once per seed, each into its own
        <output_dir>/seed_<N>/ directory. Per-seed epoch-resume is unchanged.

        Returns {seed: checkpoint_dir}.
        """
        out: dict[int, Path] = {}
        for seed in self.seeds:
            seed_dir = self.seed_output_dir(seed)
            logger.info("=== full-retrain lang=%s seed=%d → %s ===",
                        self.lang, seed, seed_dir)
            out[seed] = self.baseline.train(
                train_parquet, seed=seed, checkpoint_dir=seed_dir,
            )
        return out

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
