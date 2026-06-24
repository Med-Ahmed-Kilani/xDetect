"""
Supervised baseline: fine-tune RoBERTa-base on the English training set.

Training is reproducible (fixed seed, config-driven). The model checkpoint
and training metrics are saved to data/checkpoints/supervised_baseline/.
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
        self.model_id = cfg["model_id"]
        self.num_labels = cfg["num_labels"]
        self.max_length = cfg["max_length"]
        self.batch_size = cfg["batch_size"]
        self.lr = cfg["learning_rate"]
        self.num_epochs = cfg["num_epochs"]
        self.warmup_ratio = cfg["warmup_ratio"]
        self.weight_decay = cfg["weight_decay"]
        self.seed = cfg["seed"]
        self.checkpoint_dir = resolve_path(cfg["checkpoint_dir"])
        self.fp16 = cfg.get("fp16", False)

        _set_seed(self.seed)

        self._model = None
        self._tokenizer = None

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_tokenizer(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def train(self, train_parquet: str | Path) -> Path:
        """
        Fine-tune RoBERTa on the English training set.

        Returns the path to the saved checkpoint directory.
        """
        _set_seed(self.seed)
        self._load_tokenizer()

        df = pd.read_parquet(train_parquet)
        texts = df["text"].tolist()
        labels = df["label"].tolist()

        logger.info("Training on %d examples …", len(texts))
        dataset = TextDataset(texts, labels, self._tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, num_labels=self.num_labels
        )
        model.to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = len(loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        logger.info("Using device: %s", self.device)
        model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                            unit="batch", dynamic_ncols=True)
            for batch in progress:
                optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = model(**batch)
                out.loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += out.loss.item()
                progress.set_postfix(loss=f"{out.loss.item():.4f}")
            avg = total_loss / len(loader)
            logger.info("Epoch %d/%d — avg loss: %.4f", epoch + 1, self.num_epochs, avg)

        self._model = model
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.checkpoint_dir)
        self._tokenizer.save_pretrained(self.checkpoint_dir)
        logger.info("Checkpoint saved to %s", self.checkpoint_dir)
        return self.checkpoint_dir

    def load(self, checkpoint_dir: Optional[str | Path] = None) -> None:
        """Load a saved checkpoint."""
        ckpt = resolve_path(str(checkpoint_dir)) if checkpoint_dir else self.checkpoint_dir
        logger.info("Loading checkpoint from %s …", ckpt)
        self._tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self._model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        self._model.eval()
        self._model.to(self.device)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return probability of being machine-generated for each text."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .train() or .load() first.")
        self._model.eval()

        dataset = TextDataset(texts, [0] * len(texts), self._tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                out = self._model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
        return np.array(all_probs)

    def predict(self, texts: list[str]) -> np.ndarray:
        return (self.predict_proba(texts) >= 0.5).astype(int)

    def evaluate(self, parquet_path: str | Path) -> dict:
        """Evaluate on a parquet file and return metrics dict."""
        df = pd.read_parquet(parquet_path)
        texts = df["text"].tolist()
        labels = np.array(df["label"].tolist())
        proba = self.predict_proba(texts)
        preds = (proba >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "auroc": float(roc_auc_score(labels, proba)),
            "n": len(labels),
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
