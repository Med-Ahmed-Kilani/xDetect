"""
Trains one LoRA adapter for a given (language, n_samples) pair on top of a
frozen backbone checkpoint. Mirrors the epoch-checkpoint + resume pattern
and NaN/Inf fail-fast check from SupervisedBaseline (Month 2).
"""
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.adapters.lora_adapter import LoRAAdapter
from src.baselines.supervised_baseline import TextDataset, _set_seed
from src.config import load_config

logger = logging.getLogger(__name__)

_TRAINING_STATE = "training_state.json"
_OPTIMIZER_PT = "optimizer.pt"
_SCHEDULER_PT = "scheduler.pt"


def stratified_sample(df: pd.DataFrame, n: int | str, seed: int,
                       generator_col: str = "generator") -> pd.DataFrame:
    """
    Sample exactly n rows from df, stratified by generator_col so no single
    generator dominates a small sample. n="full" (or n >= len(df)) returns
    df shuffled in full. If a generator has fewer than n // num_groups rows,
    it simply contributes all of its rows instead of crashing.
    """
    if n == "full" or n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    groups = list(df.groupby(generator_col))
    num_groups = len(groups)
    per_group = n // num_groups
    remainder = n - per_group * num_groups

    parts = []
    for i, (_, g) in enumerate(groups):
        take = min(per_group + (1 if i < remainder else 0), len(g))
        if take > 0:
            parts.append(g.sample(n=take, random_state=seed))

    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


class AdapterTrainer:
    def __init__(self, backbone_path: str | Path, lang: str, n_samples: int | str,
                 output_dir: str | Path, cfg: Optional[dict] = None,
                 lora_cfg: Optional[dict] = None, num_labels: int = 2,
                 max_length: int = 512, seed: Optional[int] = None):
        if cfg is None:
            cfg = load_config("adapters")["adapter_training"]
        if lora_cfg is None:
            lora_cfg = load_config("adapters")["lora"]

        self.backbone_path = Path(backbone_path)
        self.lang = lang
        self.n_samples = n_samples
        self.output_dir = Path(output_dir)
        self.cfg = cfg
        self.lora_cfg = lora_cfg
        self.num_labels = num_labels
        self.max_length = max_length

        self.batch_size = cfg["batch_size"]
        self.lr = cfg["learning_rate"]
        self.num_epochs = cfg["num_epochs"]
        self.warmup_ratio = cfg["warmup_ratio"]
        self.weight_decay = cfg["weight_decay"]
        self.adam_epsilon = cfg.get("adam_epsilon", 1e-8)
        # seeds: multi-seed runs train one adapter per seed. A legacy single
        # "seed" key is accepted as a one-element list. The `seed` ctor arg
        # (used by scalability_runner) pins the active seed for this instance.
        if "seeds" in cfg:
            self.seeds = list(cfg["seeds"])
        else:
            self.seeds = [cfg["seed"]]
        self.seed = self.seeds[0] if seed is None else seed

        _set_seed(self.seed)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def seed_output_dir(self, seed: int) -> Path:
        """Per-seed adapter checkpoint sub-directory: <output_dir>/seed_<N>/."""
        return self.output_dir / f"seed_{seed}"

    def _read_training_state(self, output_dir: Path | None = None) -> dict | None:
        path = (output_dir or self.output_dir) / _TRAINING_STATE
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _write_training_state(self, epoch_completed: int,
                              output_dir: Path | None = None) -> None:
        with open((output_dir or self.output_dir) / _TRAINING_STATE, "w") as f:
            json.dump({"epoch_completed": epoch_completed,
                       "num_epochs": self.num_epochs}, f)

    def is_seed_complete(self, seed: int) -> bool:
        """True if the per-seed adapter checkpoint has all epochs recorded done."""
        state = self._read_training_state(self.seed_output_dir(seed))
        return state is not None and state["epoch_completed"] >= self.num_epochs

    def _save_epoch(self, adapter: LoRAAdapter, optimizer, scheduler,
                     epoch_completed: int, output_dir: Path | None = None) -> None:
        output_dir = output_dir or self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter.save(output_dir)
        torch.save(optimizer.state_dict(), output_dir / _OPTIMIZER_PT)
        torch.save(scheduler.state_dict(), output_dir / _SCHEDULER_PT)
        self._write_training_state(epoch_completed, output_dir)
        logger.info("Epoch %d adapter checkpoint saved to %s", epoch_completed, output_dir)

    def train(self, train_parquet: str | Path, *,
              seed: int | None = None,
              output_dir: str | Path | None = None) -> Path:
        """
        Trains the adapter, resuming from the last completed epoch if
        output_dir already has a training_state.json.

        seed / output_dir override self.seed / self.output_dir for this call —
        used by train_seeds() to train one adapter per <output_dir>/seed_<N>/.
        When omitted the instance defaults are used (unchanged single-seed
        behaviour).
        """
        seed = self.seed if seed is None else seed
        output_dir = self.output_dir if output_dir is None else Path(output_dir)

        _set_seed(seed)

        state = self._read_training_state(output_dir)
        if state is not None:
            epochs_done = state["epoch_completed"]
            if epochs_done >= self.num_epochs:
                logger.info(
                    "All %d epochs already complete for lang=%s n=%s — skipping.",
                    self.num_epochs, self.lang, self.n_samples,
                )
                return output_dir
            start_epoch = epochs_done
            logger.info("Resuming lang=%s n=%s from epoch %d/%d.",
                        self.lang, self.n_samples, start_epoch + 1, self.num_epochs)
        else:
            start_epoch = 0

        df = pd.read_parquet(train_parquet)
        sampled = stratified_sample(df, self.n_samples, seed=seed)
        logger.info("Training adapter for lang=%s on %d samples (requested %s) …",
                    self.lang, len(sampled), self.n_samples)

        texts = sampled["text"].tolist()
        labels = sampled["label"].tolist()

        adapter = LoRAAdapter(self.backbone_path, num_labels=self.num_labels,
                               cfg=self.lora_cfg)
        if start_epoch > 0:
            adapter.load(output_dir)
        else:
            adapter.build()
        trainable, fraction = adapter.trainable_parameters()
        logger.info("Adapter trainable params: %d (%.4f%%)", trainable, fraction * 100)

        model = adapter.model
        model.to(self.device)

        dataset = TextDataset(texts, labels, adapter.tokenizer, self.max_length)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        total_steps = len(loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=self.weight_decay, eps=self.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

        if start_epoch > 0:
            optimizer.load_state_dict(
                torch.load(output_dir / _OPTIMIZER_PT,
                           map_location=self.device, weights_only=False)
            )
            scheduler.load_state_dict(
                torch.load(output_dir / _SCHEDULER_PT, weights_only=False)
            )

        model.train()
        for epoch in range(start_epoch, self.num_epochs):
            total_loss = 0.0
            progress = tqdm(
                loader, desc=f"[{self.lang} n={self.n_samples}] Epoch {epoch + 1}/{self.num_epochs}",
                unit="batch", dynamic_ncols=True,
            )
            for batch in progress:
                optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = model(**batch)
                if torch.isnan(out.loss) or torch.isinf(out.loss):
                    raise RuntimeError(
                        f"Loss is {out.loss.item()} at epoch {epoch + 1}, "
                        f"batch {progress.n} (lang={self.lang}, n={self.n_samples}) — "
                        f"aborting to avoid wasting GPU time."
                    )
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += out.loss.item()
                progress.set_postfix(loss=f"{out.loss.item():.4f}")

            avg = total_loss / len(loader)
            logger.info("[%s n=%s] Epoch %d/%d — avg loss: %.4f",
                        self.lang, self.n_samples, epoch + 1, self.num_epochs, avg)
            self._save_epoch(adapter, optimizer, scheduler, epoch_completed=epoch + 1,
                             output_dir=output_dir)

        logger.info("Adapter training complete: %s", output_dir)
        return output_dir

    def train_seeds(self, train_parquet: str | Path) -> dict[int, Path]:
        """
        Train one adapter per configured seed, each into its own
        <output_dir>/seed_<N>/ directory. Per-seed epoch-resume is unchanged,
        so a partially completed multi-seed run resumes: finished seeds are
        short-circuited by train(), an interrupted seed picks up mid-run.

        Returns {seed: adapter_checkpoint_dir}.
        """
        out: dict[int, Path] = {}
        for seed in self.seeds:
            seed_dir = self.seed_output_dir(seed)
            logger.info("=== adapter lang=%s n=%s seed=%d → %s ===",
                        self.lang, self.n_samples, seed, seed_dir)
            out[seed] = self.train(train_parquet, seed=seed, output_dir=seed_dir)
        return out
