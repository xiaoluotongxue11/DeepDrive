"""Training utilities: logger, metrics, and visualization."""

import os
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MetricsBuffer:
    """Accumulates and computes running statistics for training metrics."""

    def __init__(self):
        self._data: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self._data.setdefault(key, []).append(float(value))

    def mean(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self._data.items()}

    def std(self) -> Dict[str, float]:
        return {k: float(np.std(v)) for k, v in self._data.items()}

    def clear(self) -> None:
        self._data.clear()

    def is_empty(self) -> bool:
        return len(self._data) == 0


class TrainingLogger:
    """
    Unified training logger writing to console, CSV, and optionally TensorBoard.

    Usage::

        log = TrainingLogger(log_dir="logs/task1", task_id="task1")
        log.log_epoch(epoch=1, metrics={"loss/train": 0.5, "reward/mean": 10.0})
        log.close()
    """

    def __init__(
        self,
        log_dir: str,
        task_id: str,
        use_tensorboard: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.task_id = task_id
        self.start_time = time.time()

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV file for persistent metric storage
        self._csv_path = self.log_dir / "metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None  # Initialized on first write

        # TensorBoard writer (optional dependency)
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=str(self.log_dir))
                logger.info("TensorBoard logging enabled at %s", self.log_dir)
            except ImportError:
                logger.warning("TensorBoard not available. Install tensorboard for curve visualization.")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_dir / "train.log"),
            ],
        )

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for one epoch to all outputs."""
        elapsed = time.time() - self.start_time
        metrics_with_meta = {"epoch": epoch, "elapsed_s": round(elapsed, 1), **metrics}

        # Write CSV header on first call
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(metrics_with_meta.keys())
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(metrics_with_meta)
        self._csv_file.flush()

        # TensorBoard
        if self._tb_writer is not None:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(f"{self.task_id}/{key}", value, epoch)

        # Console
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("[%s] Epoch %4d | %s", self.task_id, epoch, metric_str)

    def close(self) -> None:
        """Close all open file handles."""
        self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
