"""
Training curve analysis for DeepDrive.

Reads per-task CSV logs produced by TrainingLogger and generates:
  - Smoothed reward and loss curves (side-by-side comparison)
  - Success rate progression
  - Learning rate schedule visualization
  - Diagnostic summary with convergence statistics

Usage::

    python analysis/analyze_curves.py \
        --task1_log logs/task1/metrics.csv \
        --task3_log logs/task3/metrics.csv \
        --output_dir analysis/plots

Observations from training curves (task1=orange, task3=blue):
  - Task 1 converges faster due to simpler road topology
  - Task 3 shows higher reward variance, especially in early training
  - Both tasks benefit from the cosine LR schedule (visible as smooth
    convergence rather than abrupt plateaus)
  - Gradient clipping (task3: 0.5) visibly reduces spike frequency
    compared to task1 (1.0) at matching epochs
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def exponential_moving_average(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply EMA smoothing (equivalent to TensorBoard's smoothing slider)."""
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def convergence_epoch(values: np.ndarray, threshold: float = 0.95) -> Optional[int]:
    """Return first epoch where the metric reaches ``threshold * max(values)``."""
    target = threshold * np.max(values)
    indices = np.where(values >= target)[0]
    return int(indices[0]) if len(indices) > 0 else None


def print_summary(label: str, df: pd.DataFrame, reward_col: str = "reward/mean") -> None:
    """Print a concise diagnostics summary for one task."""
    print(f"\n{'='*50}")
    print(f"  Task: {label}")
    print(f"{'='*50}")
    print(f"  Total epochs logged : {len(df)}")

    if reward_col in df.columns:
        rewards = df[reward_col].values
        print(f"  Final reward        : {rewards[-1]:.3f}")
        print(f"  Max reward          : {np.max(rewards):.3f}  (epoch {np.argmax(rewards)})")
        print(f"  Reward std (last 20): {np.std(rewards[-20:]):.3f}")
        conv = convergence_epoch(rewards)
        print(f"  95% convergence     : epoch {conv if conv is not None else 'not reached'}")

    if "loss/train" in df.columns:
        losses = df["loss/train"].values
        print(f"  Final train loss    : {losses[-1]:.4f}")
        print(f"  Min train loss      : {np.min(losses):.4f}  (epoch {np.argmin(losses)})")

    if "success_rate" in df.columns:
        sr = df["success_rate"].values
        print(f"  Final success rate  : {sr[-1]*100:.1f}%")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curves(
    df1: pd.DataFrame,
    df3: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    smoothing: float = 0.1,
) -> None:
    """Plot a single metric for both tasks with EMA smoothing."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        print("matplotlib not installed — skipping plot generation.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Task 1 — orange (#F28E2B matches TensorBoard's default orange)
    if metric in df1.columns:
        raw1 = df1[metric].values
        smoothed1 = exponential_moving_average(raw1, alpha=smoothing)
        epochs1 = df1["epoch"].values if "epoch" in df1.columns else np.arange(len(raw1))
        ax.plot(epochs1, raw1, color="#F28E2B", alpha=0.25, linewidth=0.8)
        ax.plot(epochs1, smoothed1, color="#F28E2B", linewidth=2.0, label="Task 1")

    # Task 3 — blue (#4E79A7 matches TensorBoard's default blue)
    if metric in df3.columns:
        raw3 = df3[metric].values
        smoothed3 = exponential_moving_average(raw3, alpha=smoothing)
        epochs3 = df3["epoch"].values if "epoch" in df3.columns else np.arange(len(raw3))
        ax.plot(epochs3, raw3, color="#4E79A7", alpha=0.25, linewidth=0.8)
        ax.plot(epochs3, smoothed3, color="#4E79A7", linewidth=2.0, label="Task 3")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if "success_rate" in metric:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and compare training curves for task1 and task3."
    )
    parser.add_argument("--task1_log", type=Path, required=True,
                        help="Path to task1 metrics.csv")
    parser.add_argument("--task3_log", type=Path, required=True,
                        help="Path to task3 metrics.csv")
    parser.add_argument("--output_dir", type=Path, default=Path("analysis/plots"),
                        help="Directory to save output plots")
    parser.add_argument("--smoothing", type=float, default=0.1,
                        help="EMA smoothing factor (0=no smoothing, 1=max smoothing)")
    args = parser.parse_args()

    # Load logs
    if not args.task1_log.exists():
        print(f"ERROR: task1 log not found: {args.task1_log}", file=sys.stderr)
        sys.exit(1)
    if not args.task3_log.exists():
        print(f"ERROR: task3 log not found: {args.task3_log}", file=sys.stderr)
        sys.exit(1)

    df1 = pd.read_csv(args.task1_log)
    df3 = pd.read_csv(args.task3_log)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Print summaries
    print_summary("task1 (orange)", df1)
    print_summary("task3 (blue)", df3)

    # Generate comparison plots
    plots = [
        ("reward/mean",    "Mean Episode Reward",  "Reward Curve Comparison",      "reward_mean.png"),
        ("loss/train",     "Training Loss",         "Training Loss Comparison",     "loss_train.png"),
        ("loss/val",       "Validation Loss",       "Validation Loss Comparison",   "loss_val.png"),
        ("success_rate",   "Success Rate",          "Success Rate Comparison",      "success_rate.png"),
        ("lr",             "Learning Rate",         "Learning Rate Schedule",       "lr_schedule.png"),
    ]

    for metric, ylabel, title, filename in plots:
        if metric in df1.columns or metric in df3.columns:
            plot_curves(
                df1, df3,
                metric=metric,
                ylabel=ylabel,
                title=title,
                output_path=args.output_dir / filename,
                smoothing=args.smoothing,
            )

    print(f"\nAnalysis complete. Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
