"""
LoRA adapter wrapper around a frozen sequence-classification backbone.

Wraps a checkpoint loaded from disk (never a bare HF model id — the
backbone must already be trained; see src/adapters/adapter_trainer.py)
with a PEFT LoraConfig targeting XLM-R's query/value attention projections.
"""
import logging
from pathlib import Path
from typing import Optional

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import load_config

logger = logging.getLogger(__name__)


class LoRAAdapter:
    def __init__(self, backbone_path: str | Path, num_labels: int = 2,
                 cfg: Optional[dict] = None):
        if cfg is None:
            cfg = load_config("adapters")["lora"]
        self.cfg = cfg
        self.backbone_path = backbone_path
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None

    def _lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.cfg["r"],
            lora_alpha=self.cfg["lora_alpha"],
            lora_dropout=self.cfg["lora_dropout"],
            bias=self.cfg["bias"],
            target_modules=self.cfg["target_modules"],
            task_type=TaskType.SEQ_CLS,
        )

    def build(self):
        """Load the frozen backbone and wrap it with a fresh LoRA adapter."""
        base = AutoModelForSequenceClassification.from_pretrained(
            self.backbone_path, num_labels=self.num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_path)
        self.model = get_peft_model(base, self._lora_config())
        return self.model

    def trainable_parameters(self) -> tuple[int, float]:
        """Return (trainable_param_count, fraction_of_total)."""
        if self.model is None:
            raise RuntimeError("Call .build() or .load() first.")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        fraction = trainable / total if total else 0.0
        logger.info("Trainable params: %d / %d (%.4f%%)", trainable, total, fraction * 100)
        return trainable, fraction

    def save(self, path: str | Path) -> None:
        """Save adapter-only weights (not the frozen backbone)."""
        if self.model is None:
            raise RuntimeError("Nothing to save — call .build() first.")
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    def load(self, path: str | Path):
        """Load a previously saved adapter checkpoint onto the frozen backbone."""
        base = AutoModelForSequenceClassification.from_pretrained(
            self.backbone_path, num_labels=self.num_labels
        )
        self.model = PeftModel.from_pretrained(base, str(path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        return self.model
