"""
Main training script for DeepDrive - ContractToControl.

Supports training on task1 (lane following) and task3 (urban navigation).

Usage::

    # Train task1
    python train.py --config configs/task1.yaml

    # Train task3
    python train.py --config configs/task3.yaml

    # Resume from checkpoint
    python train.py --config configs/task3.yaml --resume logs/task3/checkpoints/epoch_100.pt

    # Override config values at CLI
    python train.py --config configs/task1.yaml training.max_epochs=100 optimizer.lr=1e-3
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_get(d: Dict, *keys, default=None):
    """Safely navigate nested dict keys."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate the policy network from config."""
    from models.policy import PolicyNetwork

    model_cfg = cfg.get("model", {})
    return PolicyNetwork(
        obs_dim=model_cfg.get("obs_dim", 256),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        action_dim=model_cfg.get("action_dim", 2),
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.1),
    )


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> AdamW:
    opt_cfg = cfg.get("optimizer", {})
    return AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 1e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    from utils.scheduler import build_scheduler as _build

    sched_cfg = cfg.get("scheduler", {})
    sched_cfg["max_epochs"] = cfg.get("training", {}).get("max_epochs", 200)
    return _build(optimizer, sched_cfg)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer, scheduler) -> int:
    """Load a checkpoint and return the epoch to resume from."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt["epoch"]) + 1


def train(cfg: Dict[str, Any], resume: str = None) -> None:
    """Main training loop."""
    from utils.logger import TrainingLogger, MetricsBuffer

    train_cfg = cfg.get("training", {})
    task_id = cfg.get("task", {}).get("id", "unknown")
    log_dir = Path(cfg.get("logging", {}).get("log_dir", "logs")) / task_id
    use_tb = cfg.get("logging", {}).get("use_tensorboard", True)

    set_seed(train_cfg.get("seed", 42))

    device_name = train_cfg.get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    # Build components
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    log = TrainingLogger(log_dir=str(log_dir), task_id=task_id, use_tensorboard=use_tb)

    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer, scheduler)
        log.log_epoch(start_epoch - 1, {"resumed": 1.0})

    max_epochs = train_cfg.get("max_epochs", 200)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    eval_interval = train_cfg.get("eval_interval", 10)
    save_interval = train_cfg.get("save_interval", 20)

    # Early stopping
    es_cfg = train_cfg.get("early_stopping", {})
    es_enabled = es_cfg.get("enabled", True)
    es_patience = int(es_cfg.get("patience", 30))
    es_min_delta = float(es_cfg.get("min_delta", 0.001))
    best_reward = -float("inf")
    patience_counter = 0

    buf = MetricsBuffer()

    for epoch in range(start_epoch, max_epochs):
        model.train()

        # ------------------------------------------------------------------ #
        # NOTE: Replace this stub with your actual data-loading and
        # environment interaction loop.
        # ------------------------------------------------------------------ #
        step_loss = _dummy_train_step(model, optimizer, device, grad_clip, cfg)
        buf.update({"loss/train": step_loss})

        # Evaluation
        if epoch % eval_interval == 0:
            val_metrics = _dummy_eval(model, device, cfg)
            epoch_metrics = {**buf.mean(), **val_metrics, "lr": scheduler.get_last_lr()[0]}
            log.log_epoch(epoch, epoch_metrics)
            buf.clear()

            # Early stopping check
            if es_enabled:
                if val_metrics.get("reward/mean", 0) > best_reward + es_min_delta:
                    best_reward = val_metrics.get("reward/mean", 0)
                    patience_counter = 0
                    save_checkpoint(
                        log_dir / "checkpoints" / "best.pt",
                        epoch, model, optimizer, scheduler, epoch_metrics,
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= es_patience:
                        log.log_epoch(epoch, {"early_stop": 1.0})
                        print(f"Early stopping at epoch {epoch}.")
                        break

        # Periodic checkpoint
        if epoch % save_interval == 0:
            save_checkpoint(
                log_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                epoch, model, optimizer, scheduler, {},
            )

        scheduler.step()

    log.close()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Placeholder train/eval steps — replace with real environment interaction
# ---------------------------------------------------------------------------

def _dummy_train_step(
    model: nn.Module,
    optimizer,
    device: torch.device,
    grad_clip: float,
    cfg: Dict[str, Any],
) -> float:
    """Stub: replace with real data loading and loss computation."""
    batch_size = cfg.get("training", {}).get("batch_size", 256)
    obs_dim = cfg.get("model", {}).get("obs_dim", 256)

    dummy_obs = {
        "image": torch.randn(batch_size, 3, 224, 224, device=device),
        "state": torch.randn(batch_size, 24, device=device),
    }
    dummy_actions = torch.randn(batch_size, 2, device=device)

    pred_actions, _ = model(dummy_obs)
    loss = nn.functional.mse_loss(pred_actions, dummy_actions)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item()


def _dummy_eval(
    model: nn.Module,
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """Stub: replace with real environment rollout evaluation."""
    model.eval()
    with torch.no_grad():
        reward = np.random.normal(loc=50.0, scale=5.0)
        success = float(np.random.binomial(n=1, p=0.6))
        val_loss = np.random.uniform(0.01, 0.5)
    return {
        "reward/mean": reward,
        "reward/std": 5.0,
        "success_rate": success,
        "loss/val": val_loss,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepDrive policy")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to task YAML config (e.g. configs/task1.yaml)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = _load_yaml(args.config)
    train(cfg, resume=args.resume)
