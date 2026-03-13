"""Tests for training utilities (scheduler and logger)."""

import math
import os
import tempfile
import pytest
import torch
from torch.optim import AdamW

from utils.scheduler import (
    WarmupCosineAnnealingLR,
    WarmupCosineAnnealingWarmRestarts,
    build_scheduler,
)
from utils.logger import MetricsBuffer, TrainingLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_optimizer():
    params = [torch.nn.Parameter(torch.zeros(1))]
    return AdamW(params, lr=1e-3)


# ---------------------------------------------------------------------------
# WarmupCosineAnnealingLR
# ---------------------------------------------------------------------------

class TestWarmupCosineAnnealingLR:
    def test_warmup_increases_lr(self, simple_optimizer):
        sched = WarmupCosineAnnealingLR(
            simple_optimizer, warmup_epochs=10, max_epochs=100, min_lr=1e-6
        )
        lrs = []
        for _ in range(10):
            lrs.append(sched.get_lr()[0])
            sched.step()
        # LR should be monotonically increasing during warm-up
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], f"LR decreased at step {i}: {lrs}"

    def test_lr_after_warmup_is_at_most_base_lr(self, simple_optimizer):
        base_lr = 1e-3
        sched = WarmupCosineAnnealingLR(
            simple_optimizer, warmup_epochs=5, max_epochs=50, min_lr=1e-6
        )
        for _ in range(5):
            sched.step()
        # After warm-up, LR should not exceed base_lr
        lr = sched.get_lr()[0]
        assert lr <= base_lr + 1e-9

    def test_final_lr_near_min(self, simple_optimizer):
        min_lr = 1e-6
        sched = WarmupCosineAnnealingLR(
            simple_optimizer, warmup_epochs=5, max_epochs=50, min_lr=min_lr
        )
        # Step to the very last epoch
        for _ in range(50):
            sched.step()
        lr = sched.get_lr()[0]
        assert lr <= min_lr * 10  # Should be near min_lr


# ---------------------------------------------------------------------------
# WarmupCosineAnnealingWarmRestarts
# ---------------------------------------------------------------------------

class TestWarmupCosineAnnealingWarmRestarts:
    def test_warmup_phase(self, simple_optimizer):
        sched = WarmupCosineAnnealingWarmRestarts(
            simple_optimizer, warmup_epochs=10, T_0=30, T_mult=2, min_lr=1e-6
        )
        lrs = []
        for _ in range(10):
            lrs.append(sched.get_lr()[0])
            sched.step()
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], f"LR should increase during warm-up"

    def test_lr_after_restart_resets_upward(self, simple_optimizer):
        base_lr = 1e-3
        sched = WarmupCosineAnnealingWarmRestarts(
            simple_optimizer, warmup_epochs=5, T_0=10, T_mult=2, min_lr=1e-6
        )
        # Run past the first cycle restart (5 warmup + 10 first cycle = epoch 15)
        for _ in range(15):
            sched.step()
        lr_after_restart = sched.get_lr()[0]
        # LR should be elevated again near base_lr after restart
        assert lr_after_restart >= base_lr * 0.5, (
            f"LR {lr_after_restart} should reset toward base_lr {base_lr} after restart"
        )


# ---------------------------------------------------------------------------
# build_scheduler factory
# ---------------------------------------------------------------------------

class TestBuildScheduler:
    def test_cosine_annealing(self, simple_optimizer):
        cfg = {
            "name": "CosineAnnealingLR",
            "warmup_epochs": 5,
            "max_epochs": 100,
            "min_lr": 1e-6,
        }
        sched = build_scheduler(simple_optimizer, cfg)
        assert isinstance(sched, WarmupCosineAnnealingLR)

    def test_cosine_warm_restarts(self, simple_optimizer):
        cfg = {
            "name": "CosineAnnealingWarmRestarts",
            "warmup_epochs": 10,
            "max_epochs": 200,
            "T_0": 50,
            "T_mult": 2,
            "min_lr": 1e-6,
        }
        sched = build_scheduler(simple_optimizer, cfg)
        assert isinstance(sched, WarmupCosineAnnealingWarmRestarts)

    def test_unknown_scheduler_raises(self, simple_optimizer):
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            build_scheduler(simple_optimizer, {"name": "NonExistent"})

    def test_invalid_warmup_raises(self, simple_optimizer):
        with pytest.raises(ValueError, match="warmup_epochs"):
            WarmupCosineAnnealingLR(simple_optimizer, warmup_epochs=-1, max_epochs=100)

    def test_max_epochs_le_warmup_raises(self, simple_optimizer):
        with pytest.raises(ValueError, match="max_epochs"):
            WarmupCosineAnnealingLR(simple_optimizer, warmup_epochs=50, max_epochs=50)


# ---------------------------------------------------------------------------
# MetricsBuffer
# ---------------------------------------------------------------------------

class TestMetricsBuffer:
    def test_mean_and_clear(self):
        buf = MetricsBuffer()
        buf.update({"loss": 1.0})
        buf.update({"loss": 3.0})
        assert buf.mean() == {"loss": pytest.approx(2.0)}
        buf.clear()
        assert buf.is_empty()

    def test_std(self):
        buf = MetricsBuffer()
        buf.update({"loss": 1.0})
        buf.update({"loss": 3.0})
        result = buf.std()
        assert "loss" in result
        assert result["loss"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TrainingLogger (basic smoke test — no TensorBoard required)
# ---------------------------------------------------------------------------

class TestTrainingLogger:
    def test_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with TrainingLogger(log_dir=tmpdir, task_id="test", use_tensorboard=False) as log:
                log.log_epoch(0, {"loss/train": 0.5, "reward/mean": 10.0})
                log.log_epoch(1, {"loss/train": 0.4, "reward/mean": 12.0})

            import csv
            csv_path = os.path.join(tmpdir, "metrics.csv")
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert float(rows[0]["loss/train"]) == pytest.approx(0.5)
            assert float(rows[1]["reward/mean"]) == pytest.approx(12.0)
