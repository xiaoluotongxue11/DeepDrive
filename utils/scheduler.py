"""Learning rate schedulers with warm-up for training stabilization."""

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing schedule with a linear warm-up phase.

    Warm-up prevents large gradient updates in early training that can destabilize
    the policy — especially important for complex tasks like task3.

    Args:
        optimizer:      The wrapped optimizer.
        warmup_epochs:  Number of epochs for the linear warm-up ramp.
        max_epochs:     Total training epochs (end of cosine cycle).
        min_lr:         Minimum learning rate at the end of the schedule.
        last_epoch:     The index of the last epoch (for resuming).

    Example::

        scheduler = WarmupCosineAnnealingLR(
            optimizer, warmup_epochs=10, max_epochs=200, min_lr=1e-6
        )
        for epoch in range(200):
            train(...)
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1.0e-6,
        last_epoch: int = -1,
    ):
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if max_epochs <= warmup_epochs:
            raise ValueError(
                f"max_epochs ({max_epochs}) must be > warmup_epochs ({warmup_epochs})"
            )
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        # base_lrs are set by _LRScheduler on first call to get_lr()
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warm-up: 0 → base_lr over warmup_epochs
            # warmup_epochs >= 1 guaranteed by __init__ when entering this branch
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing from base_lr → min_lr
            # (max_epochs - warmup_epochs) >= 1 guaranteed by __init__
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            scale = (1 + math.cos(math.pi * progress)) / 2

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine annealing with warm restarts (SGDR) plus initial linear warm-up.

    Warm restarts allow the optimizer to escape shallow local minima in the
    complex loss landscape of task3 (urban navigation), which has been shown
    to improve convergence for hard reinforcement learning tasks.

    Args:
        optimizer:      The wrapped optimizer.
        warmup_epochs:  Linear warm-up length before the first restart cycle.
        T_0:            Length of the first cosine cycle (epochs).
        T_mult:         Factor to multiply cycle length after each restart.
        min_lr:         Minimum LR at the trough of each cosine cycle.
        last_epoch:     The index of the last epoch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        T_0: int,
        T_mult: int = 2,
        min_lr: float = 1.0e-6,
        last_epoch: int = -1,
    ):
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if T_0 <= 0:
            raise ValueError(f"T_0 must be > 0, got {T_0}")
        if T_mult < 1:
            raise ValueError(f"T_mult must be >= 1, got {T_mult}")
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
            return [
                self.min_lr + (base_lr - self.min_lr) * scale
                for base_lr in self.base_lrs
            ]

        # Map epoch to position within current restart cycle
        t = epoch - self.warmup_epochs
        T_cur = self.T_0
        while t >= T_cur:
            t -= T_cur
            T_cur = T_cur * self.T_mult

        scale = (1 + math.cos(math.pi * t / T_cur)) / 2
        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


def build_scheduler(optimizer: Optimizer, cfg: dict) -> _LRScheduler:
    """
    Build a learning rate scheduler from config dict.

    Supports:
        - ``CosineAnnealingLR``             (used by task1)
        - ``CosineAnnealingWarmRestarts``   (used by task3)

    Args:
        optimizer:  The optimizer to wrap.
        cfg:        Scheduler section from the task YAML config.

    Returns:
        An instantiated _LRScheduler.

    Raises:
        ValueError: If an unsupported scheduler name is specified.
    """
    name = cfg.get("name", "CosineAnnealingLR")
    min_lr = float(cfg.get("min_lr", 1.0e-6))
    warmup = int(cfg.get("warmup_epochs", 10))
    max_epochs = int(cfg.get("max_epochs", 200))

    if name == "CosineAnnealingLR":
        return WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup,
            max_epochs=max_epochs,
            min_lr=min_lr,
        )
    if name == "CosineAnnealingWarmRestarts":
        return WarmupCosineAnnealingWarmRestarts(
            optimizer,
            warmup_epochs=warmup,
            T_0=int(cfg.get("T_0", 50)),
            T_mult=int(cfg.get("T_mult", 2)),
            min_lr=min_lr,
        )
    raise ValueError(
        f"Unsupported scheduler: '{name}'. "
        "Choose 'CosineAnnealingLR' or 'CosineAnnealingWarmRestarts'."
    )
