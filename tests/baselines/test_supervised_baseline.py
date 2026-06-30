"""
Tests for training-loop safety guards in SupervisedBaseline:
  - NaN/Inf loss triggers an immediate RuntimeError (fail-fast check)
  - Gradient clipping is applied before every optimizer step
  - adam_epsilon config knob is passed through to AdamW

No real model weights are downloaded; the model forward pass is mocked.
"""
import contextlib
import json
from pathlib import Path

import torch
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd


# ---------------------------------------------------------------------------
# Minimal config shared across tests
# ---------------------------------------------------------------------------

_BASE_CFG = {
    "model_id":       "roberta-base",
    "num_labels":     2,
    "max_length":     16,
    "batch_size":     4,
    "learning_rate":  2e-5,
    "num_epochs":     1,
    "warmup_ratio":   0.1,
    "weight_decay":   0.01,
    "seed":           42,
    "checkpoint_dir": "/tmp/test_ckpt",
}


def _fake_train_parquet(tmp_path, n: int = 8) -> str:
    path = tmp_path / "train.parquet"
    pd.DataFrame({
        "text":  [f"some training text sample {i} long enough" for i in range(n)],
        "label": [i % 2 for i in range(n)],
    }).to_parquet(path, index=False)
    return str(path)


def _make_tok_mock():
    """Tokenizer mock whose __call__ returns real tensors (uses side_effect)."""
    tok = MagicMock()
    tok.side_effect = lambda texts, **kw: {
        "input_ids":      torch.zeros(len(texts), 16, dtype=torch.long),
        "attention_mask": torch.ones(len(texts),  16, dtype=torch.long),
    }
    tok.save_pretrained = MagicMock()
    return tok


def _make_model_mock(loss_values):
    """
    Model mock whose forward pass yields a fresh loss tensor per call,
    drawn from loss_values (cycling if needed).

    A fresh tensor is required because .backward() can only be called once
    per graph node without retain_graph=True.
    """
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
    model.train = MagicMock()
    model.save_pretrained = MagicMock()
    return model


@contextlib.contextmanager
def _train_ctx(cfg, loss_values, train_path, extra_patches=()):
    """
    Patch all external dependencies so baseline.train() runs fully in-process
    with no downloads, no real optimizer type-checks, and no disk writes.

    Yields a dict of the active mocks for assertions.
    """
    tok_mock   = _make_tok_mock()
    model_mock = _make_model_mock(loss_values)
    sched_mock = MagicMock()

    with contextlib.ExitStack() as stack:
        p_tok   = stack.enter_context(
            patch("src.baselines.supervised_baseline.AutoTokenizer"))
        p_model = stack.enter_context(
            patch("src.baselines.supervised_baseline.AutoModelForSequenceClassification"))
        stack.enter_context(patch(
            "src.baselines.supervised_baseline.get_linear_schedule_with_warmup",
            return_value=sched_mock))
        stack.enter_context(patch(
            "src.baselines.supervised_baseline.resolve_path", side_effect=lambda p: p))
        # Non-resume tests don't exercise checkpointing; mock it out entirely
        # so these tests don't depend on the checkpoint dir existing on disk.
        stack.enter_context(patch(
            "src.baselines.supervised_baseline.SupervisedBaseline._save_epoch"))
        extra_mocks = [stack.enter_context(p) for p in extra_patches]

        p_tok.from_pretrained.return_value   = tok_mock
        p_model.from_pretrained.return_value = model_mock

        yield {
            "tok_cls":     p_tok,
            "model_cls":   p_model,
            "tok_mock":    tok_mock,
            "model_mock":  model_mock,
            "extra_mocks": extra_mocks,
        }


# ---------------------------------------------------------------------------
# NaN / Inf fail-fast check
# ---------------------------------------------------------------------------

class TestNanInfFailFast:
    def test_nan_loss_raises_runtime_error(self, tmp_path):
        from src.baselines.supervised_baseline import SupervisedBaseline
        baseline  = SupervisedBaseline(cfg=_BASE_CFG)
        train_path = _fake_train_parquet(tmp_path)

        with _train_ctx(_BASE_CFG, [float("nan")], train_path):
            with pytest.raises(RuntimeError, match="Loss is"):
                baseline.train(train_path)

    def test_inf_loss_raises_runtime_error(self, tmp_path):
        from src.baselines.supervised_baseline import SupervisedBaseline
        baseline  = SupervisedBaseline(cfg=_BASE_CFG)
        train_path = _fake_train_parquet(tmp_path)

        with _train_ctx(_BASE_CFG, [float("inf")], train_path):
            with pytest.raises(RuntimeError, match="Loss is"):
                baseline.train(train_path)

    def test_error_message_includes_epoch_and_batch(self, tmp_path):
        from src.baselines.supervised_baseline import SupervisedBaseline
        baseline  = SupervisedBaseline(cfg=_BASE_CFG)
        train_path = _fake_train_parquet(tmp_path)

        with _train_ctx(_BASE_CFG, [float("nan")], train_path):
            with pytest.raises(RuntimeError, match=r"epoch 1"):
                baseline.train(train_path)

    def test_normal_loss_does_not_raise(self, tmp_path):
        from src.baselines.supervised_baseline import SupervisedBaseline
        baseline  = SupervisedBaseline(cfg=_BASE_CFG)
        train_path = _fake_train_parquet(tmp_path)

        # All batches produce a healthy loss — must complete without RuntimeError
        with _train_ctx(_BASE_CFG, [0.65], train_path):
            baseline.train(train_path)   # no exception expected


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

class TestGradientClipping:
    def test_clip_grad_norm_called_before_optimizer_step(self, tmp_path):
        from src.baselines.supervised_baseline import SupervisedBaseline
        baseline  = SupervisedBaseline(cfg=_BASE_CFG)
        train_path = _fake_train_parquet(tmp_path)

        call_order = []

        def record_clip(params, max_norm, **kw):
            call_order.append("clip")
            return torch.tensor(0.8)

        mock_optimizer = MagicMock()
        mock_optimizer.step.side_effect    = lambda: call_order.append("step")
        mock_optimizer.zero_grad           = MagicMock()

        extra = [
            patch("torch.nn.utils.clip_grad_norm_", side_effect=record_clip),
            patch("torch.optim.AdamW", return_value=mock_optimizer),
        ]
        with _train_ctx(_BASE_CFG, [0.5], train_path, extra_patches=extra):
            baseline.train(train_path)

        assert "clip" in call_order, "clip_grad_norm_ was never called"
        assert "step" in call_order, "optimizer.step was never called"
        for i, event in enumerate(call_order):
            if event == "step":
                assert i > 0 and call_order[i - 1] == "clip", (
                    f"optimizer.step at index {i} not immediately preceded by clip_grad_norm_"
                )

    def test_clip_uses_max_norm_1(self, tmp_path):
        from src.baselines.supervised_baseline import SupervisedBaseline
        baseline  = SupervisedBaseline(cfg=_BASE_CFG)
        train_path = _fake_train_parquet(tmp_path)

        captured = []

        def capture_clip(params, max_norm, **kw):
            captured.append(max_norm)
            return torch.tensor(0.8)

        extra = [patch("torch.nn.utils.clip_grad_norm_", side_effect=capture_clip)]
        with _train_ctx(_BASE_CFG, [0.5], train_path, extra_patches=extra):
            baseline.train(train_path)

        assert captured, "clip_grad_norm_ was never called"
        assert all(v == 1.0 for v in captured), (
            f"Expected max_norm=1.0, got: {captured}"
        )

    def test_clip_called_once_per_batch(self, tmp_path):
        from src.baselines.supervised_baseline import SupervisedBaseline
        # 8 samples, batch_size 4 → 2 batches → clip must be called exactly twice
        cfg = {**_BASE_CFG, "batch_size": 4}
        baseline  = SupervisedBaseline(cfg=cfg)
        train_path = _fake_train_parquet(tmp_path, n=8)

        clip_count = [0]

        def count_clip(params, max_norm, **kw):
            clip_count[0] += 1
            return torch.tensor(0.8)

        extra = [patch("torch.nn.utils.clip_grad_norm_", side_effect=count_clip)]
        with _train_ctx(cfg, [0.5], train_path, extra_patches=extra):
            baseline.train(train_path)

        assert clip_count[0] == 2, (
            f"Expected clip_grad_norm_ called 2 times (one per batch), got {clip_count[0]}"
        )


# ---------------------------------------------------------------------------
# adam_epsilon config knob
# ---------------------------------------------------------------------------

class TestAdamEpsilon:
    def test_default_epsilon_is_1e8(self):
        from src.baselines.supervised_baseline import SupervisedBaseline
        baseline = SupervisedBaseline(cfg=_BASE_CFG)
        assert baseline.adam_epsilon == 1e-8

    def test_custom_epsilon_stored_on_instance(self):
        from src.baselines.supervised_baseline import SupervisedBaseline
        cfg = {**_BASE_CFG, "adam_epsilon": 1e-6}
        baseline = SupervisedBaseline(cfg=cfg)
        assert baseline.adam_epsilon == 1e-6

    def test_custom_epsilon_passed_to_adamw(self, tmp_path):
        from src.baselines.supervised_baseline import SupervisedBaseline
        cfg = {**_BASE_CFG, "adam_epsilon": 1e-6}
        baseline  = SupervisedBaseline(cfg=cfg)
        train_path = _fake_train_parquet(tmp_path)

        captured_eps = []

        real_adamw = torch.optim.AdamW

        def capture_adamw(params, lr, weight_decay, eps):
            captured_eps.append(eps)
            # Return a real AdamW so the scheduler doesn't choke
            return real_adamw(params, lr=lr, weight_decay=weight_decay, eps=eps)

        extra = [patch("torch.optim.AdamW", side_effect=capture_adamw)]
        with _train_ctx(cfg, [0.5], train_path, extra_patches=extra):
            baseline.train(train_path)

        assert captured_eps == [1e-6], (
            f"Expected AdamW to receive eps=1e-6, got: {captured_eps}"
        )


# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------
#
# The resume tests write real training_state.json / optimizer.pt / scheduler.pt
# files into tmp_path before calling train(), rather than mocking torch.load.
# This exercises the actual file-read path used during a real resume.
# torch.save is mocked to avoid pickling MagicMock state dicts.

def _setup_checkpoint(ckpt_dir: Path, epoch_completed: int, num_epochs: int) -> None:
    """Write the three files that a mid-run checkpoint produces."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "training_state.json").write_text(
        json.dumps({"epoch_completed": epoch_completed, "num_epochs": num_epochs})
    )
    # Empty dicts are valid state for mocked optimizer/scheduler load_state_dict calls
    torch.save({}, ckpt_dir / "optimizer.pt")
    torch.save({}, ckpt_dir / "scheduler.pt")


class TestResumeFromCheckpoint:
    def _make_resume_ctx(self, ckpt_dir: Path, forward_calls: list[int]):
        """
        Build an ExitStack that patches all external dependencies for a resume
        test.  forward_calls is a mutable list[int] that accumulates one count
        per model forward pass, letting the test assert how many epochs ran.

        Key differences from _train_ctx:
        - resolve_path is NOT patched (baseline already constructed with real Path)
        - Path.mkdir is NOT patched (ckpt_dir already exists)
        - torch.save IS patched (prevents pickling MagicMock state dicts)
        - torch.optim.AdamW IS patched (load_state_dict needs to accept {})
        """
        def counting_forward(**_batch):
            forward_calls[0] += 1
            out = MagicMock()
            out.loss = torch.tensor(0.5, requires_grad=True)
            return out

        model_mock = MagicMock()
        model_mock.side_effect = counting_forward
        model_mock.parameters.return_value = [torch.zeros(3, requires_grad=True)]
        model_mock.train = MagicMock()
        model_mock.save_pretrained = MagicMock()

        tok_mock = _make_tok_mock()
        sched_mock = MagicMock()
        adamw_mock = MagicMock()

        stack = contextlib.ExitStack()
        p_tok   = stack.enter_context(patch("src.baselines.supervised_baseline.AutoTokenizer"))
        p_model = stack.enter_context(patch("src.baselines.supervised_baseline.AutoModelForSequenceClassification"))
        stack.enter_context(patch("src.baselines.supervised_baseline.get_linear_schedule_with_warmup",
                                  return_value=sched_mock))
        stack.enter_context(patch("torch.optim.AdamW", return_value=adamw_mock))
        stack.enter_context(patch("torch.save"))   # prevent pickling MagicMock state dicts

        p_tok.from_pretrained.return_value   = tok_mock
        p_model.from_pretrained.return_value = model_mock

        return stack

    def test_resumes_from_correct_epoch_not_epoch_0(self, tmp_path):
        """
        After an interruption at epoch 1 of a 3-epoch run, a fresh
        SupervisedBaseline must resume from epoch 2, not epoch 0.

        Verification: count model forward passes.
          - 8 samples, batch_size 4 → 2 batches per epoch
          - If resumed correctly (2 remaining epochs): 2 × 2 = 4 forward calls
          - If started from scratch (3 epochs):        3 × 2 = 6 forward calls
        """
        from src.baselines.supervised_baseline import SupervisedBaseline

        ckpt_dir = tmp_path / "ckpt"
        _setup_checkpoint(ckpt_dir, epoch_completed=1, num_epochs=3)

        cfg = {**_BASE_CFG, "num_epochs": 3, "checkpoint_dir": str(ckpt_dir)}
        baseline = SupervisedBaseline(cfg=cfg)
        train_path = _fake_train_parquet(tmp_path, n=8)

        forward_calls = [0]
        with self._make_resume_ctx(ckpt_dir, forward_calls):
            baseline.train(train_path)

        batches_per_epoch = 8 // 4
        expected = batches_per_epoch * 2   # 2 remaining epochs
        assert forward_calls[0] == expected, (
            f"Expected {expected} forward calls (2 resumed epochs × {batches_per_epoch} batches), "
            f"got {forward_calls[0]} — training may have started from epoch 0 instead of epoch 2"
        )

    def test_skips_training_if_all_epochs_complete(self, tmp_path):
        """
        If training_state.json records epoch_completed == num_epochs, train()
        must return the checkpoint path immediately without running any batches.
        """
        from src.baselines.supervised_baseline import SupervisedBaseline

        ckpt_dir = tmp_path / "ckpt"
        _setup_checkpoint(ckpt_dir, epoch_completed=3, num_epochs=3)

        cfg = {**_BASE_CFG, "num_epochs": 3, "checkpoint_dir": str(ckpt_dir)}
        baseline = SupervisedBaseline(cfg=cfg)
        train_path = _fake_train_parquet(tmp_path, n=8)

        forward_calls = [0]
        with self._make_resume_ctx(ckpt_dir, forward_calls):
            result = baseline.train(train_path)

        assert forward_calls[0] == 0, (
            f"Expected 0 forward calls (all epochs done), got {forward_calls[0]}"
        )
        assert result == ckpt_dir

    def test_epoch_state_written_after_each_epoch(self, tmp_path):
        """
        After each epoch, training_state.json must record the epoch count
        completed so far, so an interruption between epoch N and N+1 leaves
        a consistent state.
        """
        from src.baselines.supervised_baseline import SupervisedBaseline

        ckpt_dir = tmp_path / "ckpt"
        cfg = {**_BASE_CFG, "num_epochs": 2, "checkpoint_dir": str(ckpt_dir)}
        baseline = SupervisedBaseline(cfg=cfg)
        train_path = _fake_train_parquet(tmp_path, n=4)

        forward_calls = [0]
        # Allow real writes to ckpt_dir (don't patch torch.save globally here;
        # use mock state dicts that are picklable — empty dicts are fine for
        # mocked optimizer/scheduler).
        with contextlib.ExitStack() as stack:
            p_tok   = stack.enter_context(patch("src.baselines.supervised_baseline.AutoTokenizer"))
            p_model = stack.enter_context(patch("src.baselines.supervised_baseline.AutoModelForSequenceClassification"))
            sched_mock = MagicMock()
            sched_mock.state_dict.return_value = {}   # picklable
            stack.enter_context(patch("src.baselines.supervised_baseline.get_linear_schedule_with_warmup",
                                      return_value=sched_mock))
            adamw_mock = MagicMock()
            adamw_mock.state_dict.return_value = {}   # picklable
            stack.enter_context(patch("torch.optim.AdamW", return_value=adamw_mock))

            tok_mock = _make_tok_mock()
            p_tok.from_pretrained.return_value = tok_mock

            def counting_forward(**__):
                forward_calls[0] += 1
                out = MagicMock()
                out.loss = torch.tensor(0.4, requires_grad=True)
                return out

            model_mock = MagicMock()
            model_mock.side_effect = counting_forward
            model_mock.parameters.return_value = [torch.zeros(3, requires_grad=True)]
            model_mock.train = MagicMock()
            model_mock.save_pretrained = MagicMock()
            p_model.from_pretrained.return_value = model_mock

            baseline.train(train_path)

        state = json.loads((ckpt_dir / "training_state.json").read_text())
        assert state["epoch_completed"] == 2
        assert state["num_epochs"] == 2
