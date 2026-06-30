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

logger = logging.getLogger(__name__)

_TRAINING_STATE = "training_state.json"
_OPTIMIZER_PT   = "optimizer.pt"
_SCHEDULER_PT   = "scheduler.pt"


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
        self.seed          = cfg["seed"]
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
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    # ------------------------------------------------------------------
    # Internal checkpoint helpers
    # ------------------------------------------------------------------

    def _read_training_state(self) -> dict | None:
        path = self.checkpoint_dir / _TRAINING_STATE
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _write_training_state(self, epoch_completed: int) -> None:
        with open(self.checkpoint_dir / _TRAINING_STATE, "w") as f:
            json.dump({"epoch_completed": epoch_completed,
                       "num_epochs": self.num_epochs}, f)

    def _save_epoch(self, model, tokenizer, optimizer, scheduler,
                    epoch_completed: int) -> None:
        """Persist everything needed to resume from epoch_completed + 1."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.checkpoint_dir)
        tokenizer.save_pretrained(self.checkpoint_dir)
        torch.save(optimizer.state_dict(), self.checkpoint_dir / _OPTIMIZER_PT)
        torch.save(scheduler.state_dict(), self.checkpoint_dir / _SCHEDULER_PT)
        self._write_training_state(epoch_completed)
        logger.info("Epoch %d checkpoint saved to %s", epoch_completed, self.checkpoint_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, train_parquet: str | Path) -> Path:
        """
        Fine-tune the model on train_parquet.

        Resumes from the last completed epoch if a training_state.json is
        found in checkpoint_dir.  Returns immediately (without re-training)
        if all epochs are already complete.

        Returns the path to the checkpoint directory.
        """
        _set_seed(self.seed)
        self._load_tokenizer()

        # --- Decide where to start ---
        state = self._read_training_state()
        if state is not None:
            epochs_done = state["epoch_completed"]
            if epochs_done >= self.num_epochs:
                logger.info(
                    "All %d epochs already complete — skipping training, "
                    "returning existing checkpoint at %s.",
                    self.num_epochs, self.checkpoint_dir,
                )
                return self.checkpoint_dir
            start_epoch = epochs_done
            logger.info(
                "Resuming from epoch %d/%d (checkpoint: %s).",
                start_epoch + 1, self.num_epochs, self.checkpoint_dir,
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
            generator=torch.Generator().manual_seed(self.seed),
        )

        # --- Model ---
        if start_epoch > 0:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint_dir
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, num_labels=self.num_labels
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
                torch.load(self.checkpoint_dir / _OPTIMIZER_PT,
                           map_location=self.device, weights_only=False)
            )
            scheduler.load_state_dict(
                torch.load(self.checkpoint_dir / _SCHEDULER_PT,
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
                             epoch_completed=epoch + 1)

        self._model = model
        logger.info("Training complete. Checkpoint: %s", self.checkpoint_dir)
        return self.checkpoint_dir

    def load(self, checkpoint_dir: Optional[str | Path] = None) -> None:
        """Load a saved checkpoint."""
        ckpt = resolve_path(str(checkpoint_dir)) if checkpoint_dir else self.checkpoint_dir
        logger.info("Loading checkpoint from %s …", ckpt)
        self._tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self._model     = AutoModelForSequenceClassification.from_pretrained(ckpt)
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
        Train, evaluate on test set, and save metrics to reports/.

        Returns the metrics dict.
        """
        self.train(train_parquet)
        metrics = self.evaluate(test_parquet)
        logger.info("Supervised baseline metrics: %s", metrics)

        if report_path is None:
            report_path = resolve_path("reports/supervised_baseline_en.json")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({"config": self.cfg, "metrics": metrics}, f, indent=2)
        logger.info("Report saved to %s", report_path)
        return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    baseline = SupervisedBaseline()
    baseline.run_and_save(
        train_parquet=resolve_path("data/processed/train_en.parquet"),
        test_parquet=resolve_path("data/processed/test_en.parquet"),
    )
