# DeepDrive — ContractToControl

End-to-end autonomous driving training framework supporting multiple task difficulties.

## Training Curve Review (Task 1 vs Task 3)

The training curves below compare **Task 1** (orange — lane following on straight roads) and **Task 3** (blue — urban navigation with intersections and dynamic obstacles).

### Observations

| Metric              | Task 1 (orange)                          | Task 3 (blue)                                |
|---------------------|------------------------------------------|----------------------------------------------|
| Convergence speed   | Fast (~50 epochs to 95% max reward)      | Slow (~120 epochs to 95% max reward)         |
| Reward variance     | Low (deterministic straight road)        | High (stochastic pedestrians / vehicles)     |
| Loss stability      | Stable, smooth descent                   | Oscillations during early training phase     |
| Success rate        | Reaches >80% early                       | Gradually improves, higher variance          |

### Root Causes & Fixes Applied

1. **Task 3 oscillating loss** — caused by high learning rate on complex reward landscape  
   → **Fix**: Task 3 uses `lr=1e-4` (3× lower than Task 1), stricter gradient clipping (`grad_clip=0.5`), and `CosineAnnealingWarmRestarts` to escape shallow local minima.

2. **Slow task 3 convergence** — complex multi-modal observation (camera + LiDAR + traffic state)  
   → **Fix**: Attention-based sensor fusion in `PolicyNetwork`, longer warmup (20 epochs), and a larger `patience` for early stopping (50 vs 30).

3. **High reward variance in task 3** — stochastic urban environment  
   → **Fix**: Smaller batch size (128 vs 256) for better gradient estimation, stronger data augmentation, and auxiliary value head for variance reduction.

4. **Both tasks — unstable early training**  
   → **Fix**: Linear LR warm-up before cosine annealing prevents large gradient updates in the first epochs.

## Project Structure

```
DeepDrive/
├── configs/
│   ├── base_config.yaml      # Shared hyperparameters
│   ├── task1.yaml            # Task 1 (lane following) config
│   └── task3.yaml            # Task 3 (urban navigation) config
├── models/
│   └── policy.py             # PolicyNetwork: visual + state encoder + action head
├── utils/
│   ├── logger.py             # TrainingLogger (CSV + TensorBoard)
│   └── scheduler.py          # WarmupCosineAnnealingLR / WarmRestarts
├── analysis/
│   └── analyze_curves.py     # Compare task1 vs task3 training curves
├── tests/
│   ├── test_training_utils.py
│   └── test_analyze_curves.py
├── train.py                  # Main training entry point
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Train task 1 (lane following)
python train.py --config configs/task1.yaml

# Train task 3 (urban navigation)
python train.py --config configs/task3.yaml

# Compare training curves after runs
python analysis/analyze_curves.py \
    --task1_log logs/task1/metrics.csv \
    --task3_log logs/task3/metrics.csv \
    --output_dir analysis/plots
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Hyperparameter Differences

| Parameter             | Task 1      | Task 3                         |
|-----------------------|-------------|--------------------------------|
| Learning rate         | `3e-4`      | `1e-4`                         |
| Batch size            | `256`       | `128`                          |
| Gradient clip         | `1.0`       | `0.5`                          |
| LR scheduler          | Cosine      | Cosine + Warm Restarts (T₀=50) |
| LR warmup epochs      | `10`        | `20`                           |
| Max epochs            | `200`       | `300`                          |
| Early stopping patience | `30`      | `50`                           |
| Extra sensors         | Camera only | Camera + LiDAR + traffic light |