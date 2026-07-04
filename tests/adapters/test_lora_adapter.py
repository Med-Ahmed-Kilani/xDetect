"""
Tests for LoRAAdapter (src/adapters/lora_adapter.py).

Uses a tiny locally-constructed BERT config (2 layers, hidden_size=16) saved
to a tmp dir as the "backbone checkpoint" — no downloads, no real XLM-R
weights. BERT's attention module uses the same query/key/value submodule
naming as XLM-R/RoBERTa, so target_modules=["query", "value"] matches.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig

from src.adapters.lora_adapter import LoRAAdapter

_LORA_CFG = {
    "r": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.1,
    "bias": "none",
    "target_modules": ["query", "value"],
}


@pytest.fixture(scope="module")
def tiny_backbone_dir(tmp_path_factory) -> Path:
    """Build and save a tiny 2-layer BERT-for-sequence-classification checkpoint."""
    d = tmp_path_factory.mktemp("tiny_backbone")
    config = BertConfig(
        vocab_size=200,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=64,
        num_labels=2,
    )
    model = AutoModelForSequenceClassification.from_config(config)
    model.save_pretrained(d)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(d)
    return d


class TestTrainableParameters:
    def test_trainable_params_under_2pct(self, tiny_backbone_dir):
        adapter = LoRAAdapter(tiny_backbone_dir, num_labels=2, cfg=_LORA_CFG)
        adapter.build()
        _, fraction = adapter.trainable_parameters()
        assert fraction <= 0.02, f"Expected <=2% trainable params, got {fraction:.4%}"
        assert fraction > 0.0

    def test_backbone_params_are_frozen(self, tiny_backbone_dir):
        adapter = LoRAAdapter(tiny_backbone_dir, num_labels=2, cfg=_LORA_CFG)
        model = adapter.build()
        for name, param in model.named_parameters():
            if "lora_" not in name and "classifier" not in name and "modules_to_save" not in name:
                assert not param.requires_grad, f"Backbone param {name} should be frozen"


class TestSaveLoadRoundTrip:
    def test_save_and_load_produces_identical_output(self, tiny_backbone_dir, tmp_path):
        adapter = LoRAAdapter(tiny_backbone_dir, num_labels=2, cfg=_LORA_CFG)
        adapter.build()
        adapter.model.eval()

        fixed_input = {
            "input_ids": torch.randint(0, 200, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }
        with torch.no_grad():
            original_logits = adapter.model(**fixed_input).logits

        save_dir = tmp_path / "adapter_ckpt"
        adapter.save(save_dir)

        loaded = LoRAAdapter(tiny_backbone_dir, num_labels=2, cfg=_LORA_CFG)
        loaded.load(save_dir)
        loaded.model.eval()

        with torch.no_grad():
            loaded_logits = loaded.model(**fixed_input).logits

        assert torch.allclose(original_logits, loaded_logits, atol=1e-6)


class TestLocalFilesOnly:
    """
    huggingface_hub validates repo ids and can reject an absolute local
    checkpoint path as an invalid repo id — local_files_only=True tells it
    this is a local path, not a hub lookup.
    """

    def test_build_passes_local_files_only_for_absolute_path(self, tmp_path):
        abs_backbone_path = tmp_path / "xlmr_base_checkpoint"

        with patch("src.adapters.lora_adapter.AutoModelForSequenceClassification") as p_model, \
             patch("src.adapters.lora_adapter.AutoTokenizer") as p_tok, \
             patch("src.adapters.lora_adapter.get_peft_model", return_value=MagicMock()):
            p_model.from_pretrained.return_value = MagicMock()
            p_tok.from_pretrained.return_value = MagicMock()

            adapter = LoRAAdapter(abs_backbone_path, num_labels=2, cfg=_LORA_CFG)
            adapter.build()

            p_model.from_pretrained.assert_called_once_with(
                abs_backbone_path, num_labels=2, local_files_only=True
            )
            p_tok.from_pretrained.assert_called_once_with(
                abs_backbone_path, local_files_only=True
            )

    def test_load_passes_local_files_only_for_absolute_path(self, tmp_path):
        abs_backbone_path = tmp_path / "xlmr_base_checkpoint"
        abs_adapter_path = tmp_path / "adapter_ckpt"

        with patch("src.adapters.lora_adapter.AutoModelForSequenceClassification") as p_model, \
             patch("src.adapters.lora_adapter.AutoTokenizer") as p_tok, \
             patch("src.adapters.lora_adapter.PeftModel") as p_peft:
            p_model.from_pretrained.return_value = MagicMock()
            p_tok.from_pretrained.return_value = MagicMock()
            p_peft.from_pretrained.return_value = MagicMock()

            adapter = LoRAAdapter(abs_backbone_path, num_labels=2, cfg=_LORA_CFG)
            adapter.load(abs_adapter_path)

            p_model.from_pretrained.assert_called_once_with(
                abs_backbone_path, num_labels=2, local_files_only=True
            )
            p_tok.from_pretrained.assert_called_once_with(
                str(abs_adapter_path), local_files_only=True
            )
