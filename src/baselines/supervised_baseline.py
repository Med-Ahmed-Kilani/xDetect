"""
Supervised baseline: fine-tune a sequence-classification model on a parquet
training set.  Supports per-epoch checkpointing and mid-run resume.

Resume protocol
---------------
At the start of train(), if checkpoint_dir/training_state.json exists the run
resumes from the last completed epoch.  If all epochs are already done, train()
returns immediately without re-running anything.

At the end of every epoch (not just at the very end of training) the following
are saved so a resumed run can continue cleanly:
  - model weights via model.save_pretrained(checkpoint_dir)
  - tokenizer via tokenizer.save_pretrained(checkpoint_dir)
  - optimizer state → checkpoint_dir/optimizer.pt
  - scheduler state → checkpoint_dir/scheduler.pt
  - training_state.json → {"epoch_completed": N, "num_epochs": T}
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import load_config, resolve_path
from src.eval.metrics import aggregate_seed_metrics

logger = logging.getLogger(__name__)

_TRAINING_STATE = "training_state.json"
_OPTIMIZER_PT   = "optimizer.pt"
_SCHEDULER_PT   = "scheduler.pt"


def _local_files_only_kwarg(model_id_or_path) -> dict:
    """
    huggingface_hub validates repo ids and can reject an absolute local
    checkpoint path (e.g. a warm-start checkpoint) as an invalid repo id.
    model_id is either a real HF hub id ("xlm-roberta-base") or a local
    checkpoint directory (warm start) — only the latter needs
    local_files_only=True to signal "this is a path, not a hub lookup".
    """
    return {"local_files_only": True} if Path(str(model_id_or_path)).is_dir() else {}


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class SupervisedBaseline:
    def __init__(self, cfg: Optional[dict] = None):
        if cfg is None:
            cfg = load_config("models")["supervised_baseline"]
        self.cfg = cfg
        self.model_id      = cfg["model_id"]
        self.num_labels    = cfg["num_labels"]
        self.max_length    = cfg["max_length"]
        self.batch_size    = cfg["batch_size"]
        self.lr            = cfg["learning_rate"]
        self.num_epochs    = cfg["num_epochs"]
        self.warmup_ratio  = cfg["warmup_ratio"]
        self.weight_decay  = cfg["weight_decay"]
        # seeds: a multi-seed run trains once per seed. A legacy single "seed"
        # key is accepted and treated as a one-element list. self.seed is the
        # active seed for a single-seed train() call (default: the first seed).
        if "seeds" in cfg:
            self.seeds = list(cfg["seeds"])
        else:
            self.seeds = [cfg["seed"]]
        self.seed          = self.seeds[0]
        self.checkpoint_dir = resolve_path(cfg["checkpoint_dir"])
        self.fp16          = cfg.get("fp16", False)
        self.adam_epsilon  = cfg.get("adam_epsilon", 1e-8)

        _set_seed(self.seed)

        self._model     = None
        self._tokenizer = None

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_tokenizer(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, **_local_files_only_kwarg(self.model_id)
            )

    # ------------------------------------------------------------------
    # Internal checkpoint helpers
    # ------------------------------------------------------------------

    def seed_checkpoint_dir(self, seed: int) -> Path:
        """Per-seed checkpoint sub-directory: <checkpoint_dir>/seed_<N>/."""
        return self.checkpoint_dir / f"seed_{seed}"

    def _read_training_state(self, checkpoint_dir: Path | None = None) -> dict | None:
        path = (checkpoint_dir or self.checkpoint_dir) / _TRAINING_STATE
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _write_training_state(self, epoch_completed: int,
                              checkpoint_dir: Path | None = None) -> None:
        with open((checkpoint_dir or self.checkpoint_dir) / _TRAINING_STATE, "w") as f:
            json.dump({"epoch_completed": epoch_completed,
                       "num_epochs": self.num_epochs}, f)

    def is_seed_complete(self, seed: int) -> bool:
        """True if the per-seed checkpoint has all epochs recorded as done."""
        state = self._read_training_state(self.seed_checkpoint_dir(seed))
        return state is not None and state["epoch_completed"] >= self.num_epochs

    def _save_epoch(self, model, tokenizer, optimizer, scheduler,
                    epoch_completed: int, checkpoint_dir: Path | None = None) -> None:
        """Persist everything needed to resume from epoch_completed + 1."""
        checkpoint_dir = checkpoint_dir or self.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(optimizer.state_dict(), checkpoint_dir / _OPTIMIZER_PT)
        torch.save(scheduler.state_dict(), checkpoint_dir / _SCHEDULER_PT)
        self._write_training_state(epoch_completed, checkpoint_dir)
        logger.info("Epoch %d checkpoint saved to %s", epoch_completed, checkpoint_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, train_parquet: str | Path, *,
              seed: int | None = None,
              checkpoint_dir: str | Path | None = None) -> Path:
        """
        Fine-tune the model on train_parquet.

        Resumes from the last completed epoch if a training_state.json is
        found in checkpoint_dir.  Returns immediately (without re-training)
        if all epochs are already complete.

        seed / checkpoint_dir override self.seed / self.checkpoint_dir for
        this call — used by run_multi_seed() to train one seed per
        <checkpoint_dir>/seed_<N>/ sub-directory. When omitted the instance
        defaults are used (unchanged single-seed behaviour).

        Returns the path to the checkpoint directory.
        """
        seed = self.seed if seed is None else seed
        checkpoint_dir = (self.checkpoint_dir if checkpoint_dir is None
                          else Path(checkpoint_dir))

        _set_seed(seed)
        self._load_tokenizer()

        # --- Decide where to start ---
        state = self._read_training_state(checkpoint_dir)
        if state is not None:
            epochs_done = state["epoch_completed"]
            if epochs_done >= self.num_epochs:
                logger.info(
                    "All %d epochs already complete — skipping training, "
                    "returning existing checkpoint at %s.",
                    self.num_epochs, checkpoint_dir,
                )
                return checkpoint_dir
            start_epoch = epochs_done
            logger.info(
                "Resuming from epoch %d/%d (checkpoint: %s).",
                start_epoch + 1, self.num_epochs, checkpoint_dir,
            )
        else:
            start_epoch = 0

        # --- Dataset / loader ---
        df     = pd.read_parquet(train_parquet)
        texts  = df["text"].tolist()
        labels = df["label"].tolist()
        logger.info("Training on %d examples …", len(texts))

        dataset = TextDataset(texts, labels, self._tokenizer, self.max_length)
        loader  = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        # --- Model ---
        if start_epoch > 0:
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_dir, local_files_only=True
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, num_labels=self.num_labels,
                **_local_files_only_kwarg(self.model_id)
            )
        model.to(self.device)

        # --- Optimizer and scheduler ---
        total_steps  = len(loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay, eps=self.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        if start_epoch > 0:
            optimizer.load_state_dict(
                torch.load(checkpoint_dir / _OPTIMIZER_PT,
                           map_location=self.device, weights_only=False)
            )
            scheduler.load_state_dict(
                torch.load(checkpoint_dir / _SCHEDULER_PT,
                           weights_only=False)
            )

        # --- Training loop ---
        logger.info("Using device: %s", self.device)
        model.train()
        for epoch in range(start_epoch, self.num_epochs):
            total_loss = 0.0
            progress = tqdm(
                loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                unit="batch", dynamic_ncols=True,
            )
            for batch in progress:
                optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = model(**batch)
                if torch.isnan(out.loss) or torch.isinf(out.loss):
                    raise RuntimeError(
                        f"Loss is {out.loss.item()} at epoch {epoch + 1}, "
                        f"batch {progress.n} — aborting to avoid wasting GPU time."
                    )
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += out.loss.item()
                progress.set_postfix(loss=f"{out.loss.item():.4f}")

            avg = total_loss / len(loader)
            logger.info("Epoch %d/%d — avg loss: %.4f", epoch + 1, self.num_epochs, avg)

            # Persist after every epoch so crashes don't lose work
            self._save_epoch(model, self._tokenizer, optimizer, scheduler,
                             epoch_completed=epoch + 1,
                             checkpoint_dir=checkpoint_dir)

        self._model = model
        logger.info("Training complete. Checkpoint: %s", checkpoint_dir)
        return checkpoint_dir

    def run_multi_seed(self, train_parquet: str | Path,
                       test_parquet: str | Path) -> dict[int, dict]:
        """
        Train the model once per configured seed, each into its own
        <checkpoint_dir>/seed_<N>/ directory, and evaluate every seed on
        test_parquet.

        Per-seed checkpoint + epoch-resume logic is unchanged, so a partially
        completed multi-seed run picks up where it left off: finished seeds are
        skipped by train()'s own "all epochs complete" short-circuit, and an
        interrupted seed resumes mid-run.

        Returns {seed: metrics_dict}.
        """
        per_seed: dict[int, dict] = {}
        for seed in self.seeds:
            ckpt = self.seed_checkpoint_dir(seed)
            logger.info("=== seed %d → %s ===", seed, ckpt)
            self.train(train_parquet, seed=seed, checkpoint_dir=ckpt)
            self.load(ckpt)
            per_seed[seed] = self.evaluate(test_parquet)
            logger.info("seed %d metrics: %s", seed, per_seed[seed])
        return per_seed

    def load(self, checkpoint_dir: Optional[str | Path] = None) -> None:
        """Load a saved checkpoint."""
        ckpt = resolve_path(str(checkpoint_dir)) if checkpoint_dir else self.checkpoint_dir
        logger.info("Loading checkpoint from %s …", ckpt)
        self._tokenizer = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
        self._model     = AutoModelForSequenceClassification.from_pretrained(
            ckpt, local_files_only=True
        )
        self._model.eval()
        self._model.to(self.device)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return probability of being machine-generated for each text."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .train() or .load() first.")
        self._model.eval()

        dataset = TextDataset(texts, [0] * len(texts), self._tokenizer, self.max_length)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                out = self._model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
        return np.array(all_probs)

    def predict(self, texts: list[str]) -> np.ndarray:
        return (self.predict_proba(texts) >= 0.5).astype(int)

    def evaluate(self, parquet_path: str | Path) -> dict:
        """Evaluate on a parquet file and return a metrics dict."""
        df     = pd.read_parquet(parquet_path)
        texts  = df["text"].tolist()
        labels = np.array(df["label"].tolist())
        proba  = self.predict_proba(texts)
        preds  = (proba >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1":       float(f1_score(labels, preds, zero_division=0)),
            "auroc":    float(roc_auc_score(labels, proba)),
            "n":        len(labels),
        }

    def run_and_save(self, train_parquet: str | Path,
                     test_parquet: str | Path,
                     report_path: Optional[Path] = None) -> dict:
        """
        Train once per seed, evaluate every seed on the test set, and save a
        report holding both per-seed metrics and the mean ± std aggregate.

        Returns the aggregate dict ({metric: {"mean", "std", "values"}, ...}).
        """
        per_seed = self.run_multi_seed(train_parquet, test_parquet)
        aggregate = aggregate_seed_metrics(per_seed)
        logger.info("Supervised baseline aggregate (mean ± std): %s", aggregate)

        if report_path is None:
            report_path = resolve_path("reports/supervised_baseline_en.json")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(
                {
                    "config": self.cfg,
                    "per_seed": {str(s): m for s, m in per_seed.items()},
                    "aggregate": aggregate,
                },
                f, indent=2,
            )
        logger.info("Report saved to %s", report_path)
        return aggregate


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    baseline = SupervisedBaseline()
    baseline.run_and_save(
        train_parquet=resolve_path("data/processed/train_en.parquet"),
        test_parquet=resolve_path("data/processed/test_en.parquet"),
    )
