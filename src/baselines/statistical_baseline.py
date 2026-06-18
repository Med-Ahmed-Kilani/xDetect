"""
Statistical (zero-training) baseline.

Uses GPT-2's per-token log-likelihood as a perplexity proxy.
Lower perplexity → more "fluent" → more likely machine-generated.

The decision boundary is set by calibrating on the training set scores
via a logistic threshold that maximises F1.
"""
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)


class StatisticalBaseline:
    def __init__(self, cfg: Optional[dict] = None):
        if cfg is None:
            cfg = load_config("models")["statistical_baseline"]
        self.cfg = cfg
        self.model_id = cfg["model_id"]
        self.batch_size = cfg["batch_size"]
        self.max_length = cfg["max_length"]
        self.stride = cfg["stride"]
        self.threshold = cfg["threshold"]
        self.seed = cfg["seed"]
        torch.manual_seed(self.seed)

        self._model = None
        self._tokenizer = None

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading %s for perplexity scoring …", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self._model.eval()
        self._model.to(self.device)

    def _score_text(self, text: str) -> float:
        """
        Compute the mean per-token negative log-likelihood for a single text
        using a sliding-window approach for long texts.

        Returns NLL (higher = more surprising = more human-like).
        """
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
        )
        input_ids: torch.Tensor = enc["input_ids"][0]

        if len(input_ids) == 0:
            return float("nan")

        max_len = self.max_length
        stride = self.stride

        nlls: list[float] = []
        prev_end = 0
        for begin in range(0, len(input_ids), stride):
            end = min(begin + max_len, len(input_ids))
            chunk = input_ids[begin:end].unsqueeze(0).to(self.device)
            target_len = end - prev_end
            labels = chunk.clone()
            labels[0, : chunk.size(1) - target_len] = -100

            with torch.no_grad():
                out = self._model(chunk, labels=labels)
            nlls.append(out.loss.item())
            prev_end = end
            if end == len(input_ids):
                break

        return float(np.mean(nlls))

    def score(self, texts: list[str]) -> np.ndarray:
        """Return per-text NLL scores (lower → more machine-like)."""
        self._load_model()
        scores = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info("  Scoring text %d/%d …", i, len(texts))
            scores.append(self._score_text(text))
        return np.array(scores)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Return probability of being machine-generated.

        We invert the NLL score (lower NLL → higher P(machine)) via a
        min-max normalisation, then return that as the "machine" probability.
        """
        nlls = self.score(texts)
        finite = nlls[np.isfinite(nlls)]
        if len(finite) == 0:
            return np.full(len(texts), 0.5)
        lo, hi = finite.min(), finite.max()
        if hi == lo:
            return np.full(len(texts), 0.5)
        # Invert: low NLL → high machine probability
        proba_machine = 1.0 - (nlls - lo) / (hi - lo)
        return np.clip(proba_machine, 0.0, 1.0)

    def predict(self, texts: list[str]) -> np.ndarray:
        proba = self.predict_proba(texts)
        return (proba >= self.threshold).astype(int)

    def fit_threshold(self, texts: list[str], labels: np.ndarray) -> float:
        """
        Calibrate the decision threshold on a labelled set by maximising F1.
        Updates self.threshold in place and returns the chosen value.
        """
        from sklearn.metrics import f1_score

        proba = self.predict_proba(texts)
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.01, 0.99, 99):
            preds = (proba >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = best_t
        logger.info("Calibrated threshold=%.3f (F1=%.4f)", best_t, best_f1)
        return best_t

    def run_on_dataset(self, parquet_path: str | Path) -> pd.DataFrame:
        """
        Score all texts in a parquet file. Returns the dataframe with added
        columns: `score_nll`, `proba_machine`, `pred_label`.
        """
        df = pd.read_parquet(parquet_path)
        texts = df["text"].tolist()
        nlls = self.score(texts)
        proba = self.predict_proba(texts)
        df["score_nll"] = nlls
        df["proba_machine"] = proba
        df["pred_label"] = (proba >= self.threshold).astype(int)
        return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/test_en.parquet"
    baseline = StatisticalBaseline()
    result = baseline.run_on_dataset(resolve_path(path))
    out = resolve_path("reports/statistical_baseline_scores.parquet")
    out.parent.mkdir(exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"Scores saved to {out}")
