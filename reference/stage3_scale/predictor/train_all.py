#!/usr/bin/env python3
"""Train GRU predictor on all workload patterns + combined model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictor.train import train_model
from predictor.evaluate import evaluate_model
from predictor.data import load_training_data, NormalizationParams
from predictor.gru_model import WorkloadGRU
import torch
import yaml
import time
import itertools


DATA_DIR = Path(__file__).resolve().parents[1] / 'outputs' / 'training_data'
MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'gru'

PATTERNS = {
    'A': 'diurnal',
    'B': 'bursty', 
    'C': 'batch_ramp',
    'D': 'signaling',
}


def train_per_pattern(prefer_mps: bool = False):
    """Train one GRU per workload pattern."""
    print("=" * 60)
    print("Training per-pattern models")
    print("=" * 60)

    results = {}
    for letter, name in PATTERNS.items():
        # Auto-detect training data file: prefer larger files (more data = better)
        # Sort by file size descending so we pick 168h over 24h
        candidates = sorted(
            DATA_DIR.glob(f'pattern_{letter}_*.csv'),
            key=lambda p: p.stat().st_size, reverse=True
        )
        if not candidates:
            print(f"  SKIP {name}: no training data found in {DATA_DIR}")
            continue
        csv_path = candidates[0]  # Largest file (most data)

        out_dir = MODEL_DIR / f'gru_compute-worker_{name}'
        print(f"\n--- {name} ({csv_path}) ---")
        result = train_model(
            str(csv_path), str(out_dir), prefer_mps=prefer_mps
        )
        results[name] = result

        device = 'mps' if (prefer_mps and torch.backends.mps.is_available()) else 'cpu'
        _, _, test_loader, norm = load_training_data(
            str(csv_path), h=60, k=2, batch_size=64, device=device
        )
        model = WorkloadGRU(hidden_size=64, num_layers=2, output_size=2, dropout=0.2)
        model.load_state_dict(
            torch.load(out_dir / 'model.pt', weights_only=True, map_location=device)
        )
        model.to(device)
        metrics = evaluate_model(model, test_loader, norm, device)
        print(f"  Test: {metrics.summary()}")

        config_path = out_dir / 'gru_config.yaml'
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg['evaluation'] = {
            'mape': round(metrics.mape, 2),
            'rmse': round(metrics.rmse, 4),
            'r2': round(metrics.r2, 4),
            'bias_rps': round(metrics.bias_rps, 2),
        }
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return results


def train_combined(prefer_mps: bool = True):
    """Train a single GRU on all patterns combined."""
    print("\n" + "=" * 60)
    print("Training combined model (all patterns)")
    print("=" * 60)

    out_dir = MODEL_DIR / 'gru_compute-worker_combined'
    print("(Combined training uses all 4 CSV files concatenated)")

    # We'll train on the first pattern for now, then add combined training
    # using a merged CSV approach
    candidates = sorted(DATA_DIR.glob('pattern_A_*.csv'))
    csv_path = candidates[-1] if candidates else DATA_DIR / 'pattern_A_24h.csv'
    result = train_model(str(csv_path), str(out_dir), prefer_mps=prefer_mps)
    
    return result


def run_sweep(prefer_mps: bool = True):
    """Hyperparameter sweep — slim grid on the hardest pattern (B=bursty)."""
    print("\n" + "=" * 60)
    print("Hyperparameter Sweep")
    print("=" * 60)

    candidates = sorted(DATA_DIR.glob('pattern_B_*.csv'))
    csv_path = candidates[-1] if candidates else DATA_DIR / 'pattern_B_24h.csv'
    sweep_dir = MODEL_DIR / 'sweep'

    h_values = [30, 60, 90]
    k_values = [1, 2, 4]
    hidden_sizes = [32, 64, 128]
    num_layers_vals = [1, 2]

    results = []
    for h, k, hs, nl in itertools.product(h_values, k_values, hidden_sizes, num_layers_vals):
        name = f"h{h}_k{k}_hs{hs}_nl{nl}"
        out_dir = sweep_dir / name
        print(f"\n--- Sweep: {name} ---")
        t0 = time.time()
        try:
            result = train_model(
                str(csv_path), str(out_dir),
                h=h, k=k, hidden_size=hs, num_layers=nl,
                prefer_mps=prefer_mps, epochs=100, patience=10,
            )
            result['config'] = name
            result['wall_time'] = time.time() - t0
            results.append(result)
            print(f"  val_loss={result['best_val_loss']:.6f}, "
                  f"time={result['train_time_s']:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(sweep_dir / 'sweep_results.csv', index=False)
    print(f"\nSweep complete. {len(results)} configs run. Results in {sweep_dir / 'sweep_results.csv'}")

    # Show top 3
    if len(results) > 0:
        sorted_results = sorted(results, key=lambda r: r.get('best_val_loss', float('inf')))
        print("\nTop 3 configs (by validation loss):")
        for r in sorted_results[:3]:
            print(f"  {r['config']}: val_loss={r['best_val_loss']:.6f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train GRU workload predictors')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
    parser.add_argument('--mps', action='store_true', help='Use MPS (Apple Silicon GPU) — slower for small models')
    args = parser.parse_args()

    prefer_mps = args.mps
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = train_per_pattern(prefer_mps)

    # Train combined
    # train_combined(prefer_mps)  # TODO after combined data loader

    if args.sweep:
        run_sweep(prefer_mps)

    print("\n" + "=" * 60)
    print("All training complete!")
    print(f"Models saved to: {MODEL_DIR}")
