"""
Quantile Regression (QR) — Direct quantile prediction via pinball loss.

Trains a modified GRU that outputs three quantiles (q0.05, q0.50, q0.95)
per forecast horizon simultaneously. Intervals are input-adaptive —
wider for unusual patterns, narrower for familiar ones.

Reference:
  Wen et al. (2017). Multi-Horizon Quantile Recurrent Forecaster.
  Gasthaus et al. (2019). DeepAR: Probabilistic Forecasting.
"""

from pathlib import Path
import copy
import logging
import time

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import yaml

# Resolve sibling imports
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from . import UncertaintyQuantifier, width_to_tier
from predictor.data import NormalizationParams

logger = logging.getLogger(__name__)


# ── Quantile GRU Architecture ─────────────────────────────────────────

class QuantileGRU(nn.Module):
    """GRU with 3× output head for multi-quantile prediction.

    Outputs (batch, horizon, n_quantiles) where n_quantiles=3:
      [:, :, 0] = q_low   (α/2, e.g. 0.05)
      [:, :, 1] = q_med   (0.50)
      [:, :, 2] = q_high  (1-α/2, e.g. 0.95)

    Architecture:
      GRU(hidden=64, 2 layers) → Dropout(0.2) → GRU → Linear(64, k×3)
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, output_horizon: int = 2,
                 n_quantiles: int = 3, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_horizon = output_horizon
        self.n_quantiles = n_quantiles

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_size, output_horizon * n_quantiles)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, 1) — normalized RPS window

        Returns:
            (batch, output_horizon, n_quantiles) — raw quantile predictions
            No activation on output — ReLU breaks normalized data.
        """
        out, _ = self.gru(x)
        out = out[:, -1, :]         # (batch, hidden_size)
        out = self.dropout(out)
        out = self.linear(out)       # (batch, horizon * n_quantiles)
        return out.view(-1, self.output_horizon, self.n_quantiles)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Pinball Loss ──────────────────────────────────────────────────────

def pinball_loss(y_true: Tensor, y_pred: Tensor,
                 quantiles: list[float]) -> Tensor:
    """Pinball (quantile) loss for multi-output quantile regression.

    For quantile τ, the loss is:
        L_τ(y, q̂) = { τ · (y - q̂)       if y ≥ q̂
                     { (1-τ) · (q̂ - y)   if y < q̂

    Args:
        y_true: (batch, horizon) — ground truth values
        y_pred: (batch, horizon, n_quantiles) — predicted quantiles
        quantiles: list of quantile levels (e.g. [0.05, 0.50, 0.95])

    Returns:
        scalar loss
    """
    loss = torch.tensor(0.0, device=y_pred.device)
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred[:, :, i]  # (batch, horizon)
        # pinball: max(q * errors, (q-1) * errors)
        loss += torch.mean(torch.maximum(q * errors, (q - 1) * errors))
    return loss


def monotonicity_penalty(y_pred: Tensor) -> Tensor:
    """Penalty for crossing quantiles: 0.01 * max(0, q_low - q_med).

    Args:
        y_pred: (batch, horizon, n_quantiles) where order is [low, med, high]

    Returns:
        scalar penalty (0 if non-crossing)
    """
    # Low must be ≤ med, med must be ≤ high
    q_low = y_pred[:, :, 0]
    q_med = y_pred[:, :, 1]
    q_high = y_pred[:, :, 2]

    pen_low = torch.relu(q_low - q_med)
    pen_high = torch.relu(q_med - q_high)
    return 0.01 * (pen_low.mean() + pen_high.mean())


# ── Quantile Regressor Wrapper ────────────────────────────────────────

class QuantileRegressor(UncertaintyQuantifier):
    """Direct quantile prediction via pinball loss training.

    Strengths:
      - Input-adaptive intervals (the only method with this property)
      - Single model, single forward pass
      - Captures both aleatoric and epistemic uncertainty

    Weaknesses:
      - Crossing quantiles possible (monitored + post-hoc fixed)
      - No formal coverage guarantee
      - Requires architecture modification (pinball loss)
    """

    method = 'qr'

    def __init__(self, alpha: float = 0.1,
                 h: int = 60, k: int = 2, device: str = 'cpu',
                 seed: int = 42):
        self.alpha = alpha
        self.h = h
        self.k = k
        self.device = device
        self.seed = seed

        self.quantiles = [alpha / 2, 0.50, 1 - alpha / 2]  # e.g. [0.05, 0.50, 0.95]
        self.model: QuantileGRU = None
        self.norm_params: NormalizationParams = None

        # Training config
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.2
        self.epochs = 200
        self.lr = 1e-3
        self.batch_size = 64
        self.patience = 20
        self.monotonicity_weight = 0.005  # Weight for crossing penalty

    # ── Training ────────────────────────────────────────────────────────

    def fit(self,
            train_data: tuple,
            calibration_data: tuple = None) -> 'QuantileRegressor':
        """Train QuantileGRU with pinball loss.

        Args:
            train_data: (X_train, y_train) — normalized numpy arrays
            calibration_data: Unused for QR (training split used for validation)
        """
        X_train, y_train = train_data
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)

        if X_train.ndim == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # 80/20 train/val split
        n = len(X_train)
        n_train = int(0.8 * n)

        X_tr_t = torch.from_numpy(X_train[:n_train]).to(self.device)
        y_tr_t = torch.from_numpy(y_train[:n_train]).to(self.device)
        X_val_t = torch.from_numpy(X_train[n_train:]).to(self.device)
        y_val_t = torch.from_numpy(y_train[n_train:]).to(self.device)

        logger.info("QR: Training QuantileGRU (%d params) on %d samples...",
                     QuantileGRU(hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                output_horizon=self.k).param_count,
                     n_train)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model = QuantileGRU(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_horizon=self.k,
            n_quantiles=len(self.quantiles),
            dropout=self.dropout,
        ).to(self.device)

        logger.info("QR: Model has %d parameters", self.model.param_count)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10,
        )

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            perm = torch.randperm(n_train)
            total_loss = 0.0
            n_batches = 0

            for i in range(0, n_train, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = X_tr_t[idx]
                yb = y_tr_t[idx]

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = pinball_loss(yb, pred, self.quantiles)
                if self.monotonicity_weight > 0:
                    loss += self.monotonicity_weight * monotonicity_penalty(pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            # Validation
            self.model.eval()
            with torch.inference_mode():
                val_pred = self.model(X_val_t)
                val_loss = pinball_loss(y_val_t, val_pred, self.quantiles).item()
                # Check crossing quantiles on validation
                crossing_rate = _crossing_rate(val_pred.cpu().numpy())

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("  Early stop at epoch %d (best val_loss=%.6f)",
                                 epoch + 1, best_val_loss)
                    break

            if (epoch + 1) % 20 == 0:
                logger.info("  Epoch %3d: train_loss=%.6f val_loss=%.6f crossing=%.2f%%",
                             epoch + 1, total_loss / n_batches, val_loss,
                             crossing_rate * 100)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()

        # Final crossing check
        with torch.inference_mode():
            final_pred = self.model(X_val_t).cpu().numpy()
        final_crossing = _crossing_rate(final_pred)
        logger.info("QR: Training complete. Validation crossing rate: %.2f%%",
                     final_crossing * 100)

        return self

    # ── Inference ───────────────────────────────────────────────────────

    def predict_with_uncertainty(self, history: np.ndarray) -> dict:
        """Predict with quantile-based intervals.

        Args:
            history: shape (h,) — raw RPS values
        """
        if self.model is None:
            raise RuntimeError("QR not fitted. Call fit() first.")
        if self.norm_params is None:
            raise RuntimeError("QR missing norm_params. Set before calling predict.")
        if len(history) != self.h:
            raise ValueError(f"History must have {self.h} values, got {len(history)}")

        # Normalize
        x = self.norm_params.normalize(history.astype(np.float32, copy=False))
        x_tensor = torch.from_numpy(x).to(self.device).reshape(1, self.h, 1)

        with torch.inference_mode():
            pred_norm = self.model(x_tensor).cpu().numpy().squeeze(0)  # (k, 3)

        # Denormalize quantiles
        q_low_norm = pred_norm[:, 0]
        q_med_norm = pred_norm[:, 1]
        q_high_norm = pred_norm[:, 2]

        point_forecast = self.norm_params.denormalize(q_med_norm)
        ci_lower = self.norm_params.denormalize(q_low_norm)
        ci_upper = self.norm_params.denormalize(q_high_norm)

        # Enforce monotonicity (fix crossing quantiles post-hoc)
        has_crossing = False
        for step in range(self.k):
            if ci_lower[step] > point_forecast[step]:
                ci_lower[step], point_forecast[step] = point_forecast[step], ci_lower[step]
                has_crossing = True
            if point_forecast[step] > ci_upper[step]:
                point_forecast[step], ci_upper[step] = ci_upper[step], point_forecast[step]
                has_crossing = True

        # Non-negative
        np.maximum(ci_lower, 0.0, out=ci_lower)
        np.maximum(point_forecast, 0.0, out=point_forecast)
        np.maximum(ci_upper, 0.0, out=ci_upper)

        # Confidence score: normalized interval width
        mean_rate = float(np.mean(point_forecast))
        if mean_rate > 1e-6:
            confidence_score = float(np.mean(ci_upper - ci_lower) / mean_rate)
        else:
            confidence_score = 0.0

        tier = width_to_tier(confidence_score)

        return {
            'point_forecast': point_forecast,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_score': confidence_score,
            'tier': tier,
            'method': 'qr',
            'metadata': {
                'quantiles': self.quantiles,
                'q_low': ci_lower.tolist(),
                'q_med': point_forecast.tolist(),
                'q_high': ci_upper.tolist(),
                'has_crossing': has_crossing,
            }
        }

    # ── Evaluation ──────────────────────────────────────────────────────

    def evaluate_coverage(self, test_data: tuple) -> dict:
        """Evaluate empirical coverage on test data."""
        X_test, y_test = test_data
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)

        y_test_rps = np.array([
            self.norm_params.denormalize(y_test[i])
            for i in range(len(y_test))
        ])

        coverages = []
        widths = []
        mapes = []
        tiers = {1: 0, 2: 0, 3: 0}
        times_ms = []
        n_crossing = 0

        for i in range(len(X_test)):
            history_raw = self.norm_params.denormalize(X_test[i].flatten())

            t0 = time.time()
            pred = self.predict_with_uncertainty(history_raw)
            times_ms.append((time.time() - t0) * 1000)

            if pred['metadata']['has_crossing']:
                n_crossing += 1

            y_true = y_test_rps[i]
            for step in range(self.k):
                in_interval = float(
                    pred['ci_lower'][step] <= y_true[step] <= pred['ci_upper'][step]
                )
                coverages.append(in_interval)
                widths.append(float(pred['ci_upper'][step] - pred['ci_lower'][step]))
                mapes.append(
                    float(abs(y_true[step] - pred['point_forecast'][step]) /
                          (y_true[step] + 1e-6))
                )

            tiers[pred['tier']] += 1

        return {
            'method': 'qr',
            'empirical_coverage': float(np.mean(coverages)),
            'target_coverage': 1.0 - self.alpha,
            'mean_interval_width': float(np.mean(widths)),
            'mean_mape': float(np.mean(mapes)),
            'mean_inference_ms': float(np.mean(times_ms)),
            'tier_distribution': tiers,
            'n_test_samples': len(X_test),
            'crossing_rate': n_crossing / len(X_test),
            'quantiles': self.quantiles,
        }

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.model:
            torch.save(self.model.state_dict(), output_dir / 'quantile_model.pt')

        config = {
            'method': 'qr',
            'alpha': self.alpha,
            'h': self.h,
            'k': self.k,
            'quantiles': self.quantiles,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'seed': self.seed,
        }
        if self.norm_params:
            config['normalization'] = {
                'mu': self.norm_params.mu,
                'sigma': self.norm_params.sigma,
            }
        with open(output_dir / 'qr_config.yaml', 'w') as f:
            yaml.dump(config, f)

        logger.info("QR saved to %s", output_dir)

    @classmethod
    def load(cls, model_dir: str, device: str = 'cpu') -> 'QuantileRegressor':
        model_dir = Path(model_dir)
        with open(model_dir / 'qr_config.yaml') as f:
            config = yaml.safe_load(f)

        qr = cls(
            alpha=config['alpha'],
            h=config['h'],
            k=config['k'],
            device=device,
            seed=config['seed'],
        )
        qr.quantiles = config['quantiles']
        qr.hidden_size = config['hidden_size']
        qr.num_layers = config['num_layers']
        qr.dropout = config['dropout']

        if 'normalization' in config:
            qr.norm_params = NormalizationParams(
                mu=config['normalization']['mu'],
                sigma=config['normalization']['sigma'],
            )

        # Load model
        qr.model = QuantileGRU(
            input_size=1,
            hidden_size=qr.hidden_size,
            num_layers=qr.num_layers,
            output_horizon=qr.k,
            n_quantiles=len(qr.quantiles),
            dropout=qr.dropout,
        ).to(device)
        qr.model.load_state_dict(
            torch.load(model_dir / 'quantile_model.pt',
                      weights_only=True, map_location=device)
        )
        qr.model.eval()

        logger.info("QR loaded from %s", model_dir)
        return qr


# ── Helpers ────────────────────────────────────────────────────────────

def _crossing_rate(preds: np.ndarray) -> float:
    """Fraction of predictions where q_low > q_med or q_med > q_high.

    Args:
        preds: (N, k, 3) — batch of quantile predictions

    Returns:
        Fraction of (sample, step) pairs with crossing quantiles
    """
    q_low = preds[:, :, 0]
    q_med = preds[:, :, 1]
    q_high = preds[:, :, 2]
    crossing = (q_low > q_med) | (q_med > q_high)
    return float(crossing.mean())
