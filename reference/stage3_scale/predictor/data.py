"""Data loading pipeline for GRU workload predictor."""

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass
from typing import Optional


@dataclass
class NormalizationParams:
    """mu, sigma for z-score normalization — saved alongside the model."""
    mu: float
    sigma: float

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sigma

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.sigma + self.mu


def _validate_window_sizes(n_rows: int, h: int, k: int) -> None:
    min_rows = h + k + 1
    if n_rows < min_rows:
        raise ValueError(
            f"Dataset has {n_rows} rows but need at least {min_rows} "
            f"for h={h}, k={k}. Increase training data or reduce window sizes."
        )


def load_training_data(
    csv_path: str,
    h: int = 60,
    k: int = 2,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> tuple:
    """Split chronologically 60/20/20; h and k count 30-second intervals.

    Return (train_loader, val_loader, test_loader, norm_params).
    """
    df = pd.read_csv(csv_path)
    _validate_window_sizes(len(df), h, k)

    # Normalize per-pattern
    mu = float(df['rps'].mean())
    sigma = float(df['rps'].std())
    if sigma == 0:
        sigma = 1.0  # Edge case: constant workload
    norm = NormalizationParams(mu=mu, sigma=sigma)
    rps_norm = (df['rps'].values - mu) / sigma

    # Sliding window construction: predict k steps ahead from h-step history
    X_list, y_list = [], []
    for i in range(len(rps_norm) - h - k + 1):
        X_list.append(rps_norm[i:i + h])
        y_list.append(rps_norm[i + h:i + h + k])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Reshape: (n_samples, seq_len, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    # y shape: (n_samples, k) — already correct

    # Chronological split (60/20/20) — no shuffling to preserve temporal order
    n = len(X)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    dev = device or 'cpu'
    train_ds = TensorDataset(torch.tensor(X[:train_end], device=dev),
                             torch.tensor(y[:train_end], device=dev))
    val_ds = TensorDataset(torch.tensor(X[train_end:val_end], device=dev),
                           torch.tensor(y[train_end:val_end], device=dev))
    test_ds = TensorDataset(torch.tensor(X[val_end:], device=dev),
                            torch.tensor(y[val_end:], device=dev))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, norm


def load_live_history(csv_path: str, norm: NormalizationParams, h: int = 60) -> np.ndarray:
    """Return normalized history of shape (1, h, 1)."""
    df = pd.read_csv(csv_path)
    if len(df) < h:
        raise ValueError(f"Need at least {h} rows, got {len(df)}")
    last_h = df['rps'].values[-h:]
    return norm.normalize(last_h).reshape(1, h, 1).astype(np.float32)
