"""
Tests for AdapterTrainer (src/adapters/adapter_trainer.py):
  - stratified_sample: balanced sampling, "full" passthrough, small-generator
    edge case that must not crash
  - NaN/Inf fail-fast during training
  - epoch-level checkpoint + resume

The model forward pass is mocked (same style as test_supervised_baseline.py)
so no real weights are downloaded.
"""
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock, patch

from src.adapters.adapter_trainer import AdapterTrainer, stratified_sample

_BASE_CFG = {
    "num_epochs": 1,
    "learning_rate": 3e-4,
    "batch_size": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "seed": 42,
}
_LORA_CFG = {
    "r": 4, "lora_alpha": 8, "lora_dropout": 0.1,
    "bias": "none", "target_modules": ["query", "value"],
}


# ---------------------------------------------------------------------------
# stratified_sample
# ---------------------------------------------------------------------------

def _make_df(n_per_gen: dict[str, int]) -> pd.DataFrame:
    rows = []
    for gen, n in n_per_gen.items():
        for i in range(n):
            rows.append({"text": f"{gen}-{i}", "label": 0 if gen == "human" else 1,
                        "generator": gen})
    return pd.DataFrame(rows)


class TestStratifiedSample:
    def test_full_returns_all_rows_shuffled(self):
        df = _make_df({"human": 10, "gpt": 10})
        result = stratified_sample(df, "full", seed=42)
        assert len(result) == 20
        assert set(result["text"]) == set(df["text"])

    def test_n_greater_than_len_returns_all_rows(self):
        df = _make_df({"human": 5, "gpt": 5})
        result = stratified_sample(df, 1000, seed=42)
        assert len(result) == 10

    def test_exact_n_returned_when_data_sufficient(self):
        df = _make_df({"human": 50, "gpt": 50, "llama": 50, "claude": 50})
        result = stratified_sample(df, 40, seed=42)
        assert len(result) == 40

    def test_no_single_generator_dominates(self):
        df = _make_df({"human": 100, "gpt": 100, "llama": 100, "claude": 100})
        result = stratified_sample(df, 40, seed=42)
        counts = result["generator"].value_counts()
        assert counts.max() - counts.min() <= 1

    def test_deterministic_with_fixed_seed(self):
        df = _make_df({"human": 50, "gpt": 50})
        r1 = stratified_sample(df, 20, seed=42)
        r2 = stratified_sample(df, 20, seed=42)
        assert list(r1["text"]) == list(r2["text"])

    def test_generator_with_fewer_than_needed_does_not_crash(self):
        # "rare" generator only has 2 rows, far fewer than n // num_groups
        df = _make_df({"human": 100, "gpt": 100, "rare": 2})
        result = stratified_sample(df, 30, seed=42)
        assert len(result) > 0
        rare_count = (result["generator"] == "rare").sum()
        assert rare_count <= 2


# ---------------------------------------------------------------------------
# Training loop safety (NaN/Inf fail-fast) + resume
# ---------------------------------------------------------------------------

def _fake_train_parquet(tmp_path, n: int = 8) -> str:
    path = tmp_path / "train_de.parquet"
    pd.DataFrame({
        "text": [f"sample text {i}" for i in range(n)],
        "label": [i % 2 for i in range(n)],
        "generator": ["human" if i % 2 == 0 else "gpt" for i in range(n)],
    }).to_parquet(path, index=False)
    return str(path)


def _make_tok_mock():
    tok = MagicMock()
    tok.side_effect = lambda texts, **kw: {
        "input_ids": torch.zeros(len(texts), 8, dtype=torch.long),
        "attention_mask": torch.ones(len(texts), 8, dtype=torch.long),
    }
    return tok


def _make_model_mock(loss_values):
    values = list(loss_values)
    call_count = [0]

    def forward(**batch):
        v = values[min(call_count[0], len(values) - 1)]
        call_count[0] += 1
        out = MagicMock()
        out.loss = torch.tensor(v, requires_grad=True)
        return out

    model = MagicMock()
    model.side_effect = forward
    model.parameters.return_value = [torch.zeros(3, requires_grad=True)]
    model.to = MagicMock(return_value=model)
    model.train = MagicMock()
    return model


def _make_adapter_mock(loss_values, forward_calls=None):
    """Mocks LoRAAdapter so .build()/.load() set .model/.tokenizer directly."""
    tok_mock = _make_tok_mock()
    model_mock = _make_model_mock(loss_values)
    if forward_calls is not None:
        real_side_effect = model_mock.side_effect

        def counting(**batch):
            forward_calls[0] += 1
            return real_side_effect(**batch)
        model_mock.side_effect = counting

    adapter_mock = MagicMock()
    adapter_mock.model = model_mock
    adapter_mock.tokenizer = tok_mock
    adapter_mock.build.return_value = model_mock
    adapter_mock.load.return_value = model_mock
    adapter_mock.trainable_parameters.return_value = (10, 0.01)
    adapter_mock.save = MagicMock()
    return adapter_mock


@contextlib.contextmanager
def _train_ctx(loss_values, extra_patches=(), forward_calls=None):
    adapter_mock = _make_adapter_mock(loss_values, forward_calls=forward_calls)
    sched_mock = MagicMock()

    with contextlib.ExitStack() as stack:
        p_adapter_cls = stack.enter_context(
            patch("src.adapters.adapter_trainer.LoRAAdapter", return_value=adapter_mock))
        stack.enter_context(patch(
            "src.adapters.adapter_trainer.get_linear_schedule_with_warmup",
            return_value=sched_mock))
        stack.enter_context(patch(
            "src.adapters.adapter_trainer.AdapterTrainer._save_epoch"))
        extra_mocks = [stack.enter_context(p) for p in extra_patches]
        yield {"adapter_cls": p_adapter_cls, "adapter_mock": adapter_mock,
              "extra_mocks": extra_mocks}


class TestNanInfFailFast:
    def test_nan_loss_raises_runtime_error(self, tmp_path):
        train_path = _fake_train_parquet(tmp_path)
        trainer = AdapterTrainer(
            backbone_path=tmp_path / "backbone", lang="de", n_samples="full",
            output_dir=tmp_path / "ckpt", cfg=_BASE_CFG, lora_cfg=_LORA_CFG,
        )
        with _train_ctx([float("nan")]):
            with pytest.raises(RuntimeError, match="Loss is"):
                trainer.train(train_path)

    def test_inf_loss_raises_runtime_error(self, tmp_path):
        train_path = _fake_train_parquet(tmp_path)
        trainer = AdapterTrainer(
            backbone_path=tmp_path / "backbone", lang="ar", n_samples=100,
            output_dir=tmp_path / "ckpt", cfg=_BASE_CFG, lora_cfg=_LORA_CFG,
        )
        with _train_ctx([float("inf")]):
            with pytest.raises(RuntimeError, match="Loss is"):
                trainer.train(train_path)

    def test_normal_loss_completes_without_raising(self, tmp_path):
        train_path = _fake_train_parquet(tmp_path)
        trainer = AdapterTrainer(
            backbone_path=tmp_path / "backbone", lang="de", n_samples="full",
            output_dir=tmp_path / "ckpt", cfg=_BASE_CFG, lora_cfg=_LORA_CFG,
        )
        with _train_ctx([0.5]):
            result = trainer.train(train_path)
        assert result == trainer.output_dir


class TestResumeFromCheckpoint:
    def _setup_checkpoint(self, ckpt_dir: Path, epoch_completed: int, num_epochs: int):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "training_state.json").write_text(
            json.dumps({"epoch_completed": epoch_completed, "num_epochs": num_epochs})
        )
        torch.save({}, ckpt_dir / "optimizer.pt")
        torch.save({}, ckpt_dir / "scheduler.pt")

    def test_resumes_from_correct_epoch(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        self._setup_checkpoint(ckpt_dir, epoch_completed=1, num_epochs=3)
        train_path = _fake_train_parquet(tmp_path, n=8)

        cfg = {**_BASE_CFG, "num_epochs": 3, "batch_size": 4}
        trainer = AdapterTrainer(
            backbone_path=tmp_path / "backbone", lang="de", n_samples="full",
            output_dir=ckpt_dir, cfg=cfg, lora_cfg=_LORA_CFG,
        )

        forward_calls = [0]
        adamw_mock = MagicMock()
        extra = [
            patch("torch.optim.AdamW", return_value=adamw_mock),
            patch("torch.save"),
        ]
        with _train_ctx([0.5], extra_patches=extra, forward_calls=forward_calls):
            trainer.train(train_path)

        batches_per_epoch = 8 // 4
        assert forward_calls[0] == batches_per_epoch * 2  # 2 remaining epochs

    def test_skips_training_if_all_epochs_complete(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        self._setup_checkpoint(ckpt_dir, epoch_completed=3, num_epochs=3)
        train_path = _fake_train_parquet(tmp_path, n=8)

        cfg = {**_BASE_CFG, "num_epochs": 3}
        trainer = AdapterTrainer(
            backbone_path=tmp_path / "backbone", lang="de", n_samples="full",
            output_dir=ckpt_dir, cfg=cfg, lora_cfg=_LORA_CFG,
        )

        forward_calls = [0]
        with _train_ctx([0.5], forward_calls=forward_calls):
            result = trainer.train(train_path)

        assert forward_calls[0] == 0
        assert result == ckpt_dir
