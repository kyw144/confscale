"""
Evaluation metrics for GRU workload predictor.

Computes MAPE, RMSE, R², and prediction bias on normalized or denormalized data.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from .data import NormalizationParams


@dataclass
class EvalMetrics:
    mape: float       # Mean Absolute Percentage Error (%)
    rmse: float       # Root Mean Squared Error (normalized)
    rmse_rps: float   # Root Mean Squared Error (in RPS units)
    r2: float         # Coefficient of determination
    bias: float       # Mean(ŷ − y) — systematic bias
    bias_rps: float   # Mean(ŷ − y) in RPS units
    n_samples: int

    def summary(self) -> str:
        return (
            f"MAPE={self.mape:.1f}%  RMSE={self.rmse:.4f}  "
            f"R²={self.r2:.4f}  Bias={self.bias:.4f}  N={self.n_samples}"
        )


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    norm: Optional[NormalizationParams] = None,
    device: str = 'cpu',
) -> EvalMetrics:
    """Evaluate a trained model on a test DataLoader.

    Args:
        model: Trained WorkloadGRU
        test_loader: DataLoader with (X, y) tensors
        norm: NormalizationParams for denormalizing to RPS
        device: torch device string

    Returns:
        EvalMetrics with MAPE, RMSE, R², bias
    """
    model.eval()
    criterion = nn.MSELoss()

    all_y = []
    all_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(device))
            all_y.append(y_batch.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y = np.concatenate(all_y).flatten()
    y_hat = np.concatenate(all_pred).flatten()

    return _compute_metrics(y, y_hat, norm)


def _compute_metrics(
    y: np.ndarray,
    y_hat: np.ndarray,
    norm: Optional[NormalizationParams] = None,
) -> EvalMetrics:
    """Compute all metrics from raw (normalized) predictions."""
    n = len(y)

    # RMSE (normalized)
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))

    # R² — guard against division by ~0 (constant test data)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-10:
        r2 = 0.0  # Constant ground truth — R² is undefined
    else:
        r2 = float(1 - ss_res / ss_tot)

    # Bias (normalized)
    bias = float(np.mean(y_hat - y))

    # If normalization params available, compute denormalized metrics
    if norm is not None:
        y_rps = norm.denormalize(y)
        y_hat_rps = norm.denormalize(y_hat)
        rmse_rps = float(np.sqrt(np.mean((y_rps - y_hat_rps) ** 2)))
        bias_rps = float(np.mean(y_hat_rps - y_rps))

        # MAPE on denormalized data (clip y_rps to avoid division by zero)
        y_rps_safe = np.maximum(np.abs(y_rps), 0.1)
        mape = float(np.mean(np.abs((y_rps - y_hat_rps) / y_rps_safe)) * 100)
    else:
        rmse_rps = rmse  # Fallback
        bias_rps = bias
        mape = float(np.mean(np.abs((y - y_hat) / np.maximum(np.abs(y), 0.01))) * 100)

    return EvalMetrics(
        mape=mape,
        rmse=rmse,
        rmse_rps=rmse_rps,
        r2=r2,
        bias=bias,
        bias_rps=bias_rps,
        n_samples=n,
    )
