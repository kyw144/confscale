#!/usr/bin/env python3
"""Standalone hyperparameter sweep — only runs sweep, no per-pattern re-training."""
import sys, time, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from predictor.train import train_model
from predictor.evaluate import evaluate_model
from predictor.data import load_training_data
from predictor.gru_model import WorkloadGRU
import torch

DATA_DIR = Path(__file__).resolve().parents[1] / 'outputs' / 'training_data'
SWEEP_DIR = Path(__file__).resolve().parents[1] / 'models' / 'gru' / 'sweep'
CSV_PATH = DATA_DIR / 'pattern_B_24h.csv'  # Hardest pattern = bursty

# Grid from the plan
h_values = [30, 60, 90]
k_values = [1, 2, 4]
hidden_sizes = [32, 64, 128]
num_layers_vals = [1, 2]

SWEEP_DIR.mkdir(parents=True, exist_ok=True)

results = []
total = len(h_values) * len(k_values) * len(hidden_sizes) * len(num_layers_vals)
i = 0

for h, k, hs, nl in itertools.product(h_values, k_values, hidden_sizes, num_layers_vals):
    i += 1
    name = f"h{h}_k{k}_hs{hs}_nl{nl}"
    out_dir = SWEEP_DIR / name
    print(f"[{i}/{total}] {name} ... ", end='', flush=True)
    t0 = time.time()
    try:
        result = train_model(
            str(CSV_PATH), str(out_dir),
            h=h, k=k, hidden_size=hs, num_layers=nl,
            prefer_mps=False, epochs=100, patience=10,
        )
        result['config'] = name
        result['h'] = h
        result['k'] = k
        result['hidden_size'] = hs
        result['num_layers'] = nl
        result['wall_time'] = time.time() - t0
        results.append(result)
        print(f"val_loss={result['best_val_loss']:.6f}, time={result['train_time_s']:.1f}s")
    except Exception as e:
        print(f"FAILED: {e}")

# Save results
if results:
    df = pd.DataFrame(results)
    df.to_csv(SWEEP_DIR / 'sweep_results.csv', index=False)
    print(f"\n{len(results)}/{total} configs completed. Results saved.")

    # Top 5 by validation loss
    sorted_results = sorted(results, key=lambda r: r.get('best_val_loss', float('inf')))
    print("\nTop 5 configs:")
    for r in sorted_results[:5]:
        print(f"  {r['config']}: val_loss={r['best_val_loss']:.6f}, time={r['train_time_s']:.1f}s")
else:
    print("\nNo results collected.")
