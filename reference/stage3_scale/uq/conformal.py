"""Split-conformal prediction; coverage guarantees require exchangeability.

References: Vovk, Gammerman & Shafer (2005); Lin, Trivedi & Sun (2022).
"""

from pathlib import Path
import logging
import time

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from . import UncertaintyQuantifier, width_to_tier
from predictor.gru_model import WorkloadGRU
from predictor.data import NormalizationParams

logger = logging.getLogger(__name__)


class SplitConformal(UncertaintyQuantifier):
    """Inductive (split) conformal prediction for time series forecasting."""

    method = 'scp'

    def __init__(self, alpha: float = 0.1,
                 h: int = 60, k: int = 2, device: str = 'cpu'):
        self.alpha = alpha
        self.h = h
        self.k = k
        self.device = device

        self.base_model: WorkloadGRU = None
        self.norm_params: NormalizationParams = None

        self.q_hat: np.ndarray = None  # shape (k,)

        self.local_weighting: bool = False
        self._local_residual_std: float = 0.0

    def fit(self,
            train_data: tuple,
            calibration_data: tuple = None) -> 'SplitConformal':
        """Calibrate conformal quantiles on held-out calibration data."""
        X_train, y_train = train_data
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)

        # Split train_data if no separate calibration set
        if calibration_data is None:
            n = len(X_train)
            n_cal = int(0.2 * n)
            X_cal, y_cal = X_train[-n_cal:], y_train[-n_cal:]
            X_tr = X_train[:-n_cal]
            y_tr = y_train[:-n_cal]
        else:
            X_cal, y_cal = calibration_data
            X_cal = np.asarray(X_cal, dtype=np.float32)
            y_cal = np.asarray(y_cal, dtype=np.float32)
            X_tr, y_tr = X_train, y_train

        if X_tr.ndim == 2:
            X_tr = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
        if X_cal.ndim == 2:
            X_cal = X_cal.reshape(X_cal.shape[0], X_cal.shape[1], 1)

        logger.info("SCP: Training base GRU on %d samples...", len(X_tr))
        from .bootstrap import _train_gru

        self.base_model = WorkloadGRU(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=self.k,
            dropout=0.2,
        ).to(self.device)

        self.base_model = _train_gru(
            model=self.base_model,
            X=X_tr,
            y=y_tr,
            epochs=200,
            lr=1e-3,
            batch_size=64,
            patience=20,
            device=self.device,
            seed=42,
        )

        logger.info("SCP: Calibrating on %d samples...", len(X_cal))
        X_cal_t = torch.from_numpy(X_cal).to(self.device)
        y_cal_t = torch.from_numpy(y_cal).to(self.device)

        self.base_model.eval()
        with torch.inference_mode():
            preds = self.base_model(X_cal_t).cpu().numpy()  # (N_cal, k)

        scores = np.abs(y_cal - preds)  # (N_cal, k)

        n_cal = len(scores)
        # Conformal quantile with finite-sample correction
        q_index = int(np.ceil((1 - self.alpha) * (n_cal + 1))) - 1
        q_index = min(q_index, n_cal - 1)

        self.q_hat = np.array([
            np.sort(scores[:, step])[q_index]
            for step in range(self.k)
        ])

        logger.info("SCP: q_hat = %s (coverage target: %.0f%%)",
                     np.array2string(self.q_hat, precision=2),
                     (1 - self.alpha) * 100)

        self._local_residual_std = float(np.std(scores))

        covered = (y_cal >= preds - self.q_hat) & (y_cal <= preds + self.q_hat)
        emp_coverage = float(covered.mean())
        logger.info("SCP: Calibration coverage = %.2f%% (target: %.0f%%)",
                     emp_coverage * 100, (1 - self.alpha) * 100)

        return self

    def predict_with_uncertainty(self, history: np.ndarray) -> dict:
        """Predict with conformal intervals."""
        if self.base_model is None or self.q_hat is None:
            raise RuntimeError("SCP not calibrated. Call fit() first.")
        if self.norm_params is None:
            raise RuntimeError("SCP missing norm_params. Set before calling predict.")
        if len(history) != self.h:
            raise ValueError(f"History must have {self.h} values, got {len(history)}")

        # Normalize and predict
        x = self.norm_params.normalize(history.astype(np.float32, copy=False))
        x_tensor = torch.from_numpy(x).to(self.device).reshape(1, self.h, 1)

        with torch.inference_mode():
            pred_norm = self.base_model(x_tensor).cpu().numpy().flatten()

        point_forecast = self.norm_params.denormalize(pred_norm)
        np.maximum(point_forecast, 0.0, out=point_forecast)

        # q_hat is normalized; multiply by sigma to obtain RPS widths.
        ci_lower = point_forecast - self.norm_params.sigma * self.q_hat
        ci_upper = point_forecast + self.norm_params.sigma * self.q_hat

        # No negative rates
        np.maximum(ci_lower, 0.0, out=ci_lower)
        np.maximum(ci_upper, 0.0, out=ci_upper)

        # Confidence score: normalized interval width
        mean_rate = float(np.mean(point_forecast))
        interval_width = float(np.mean(ci_upper - ci_lower))
        if mean_rate > 1e-6:
            confidence_score = interval_width / mean_rate
        else:
            confidence_score = 0.0

        tier = width_to_tier(confidence_score)

        return {
            'point_forecast': point_forecast,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_score': confidence_score,
            'tier': tier,
            'method': 'scp',
            'metadata': {
                'q_hat': self.q_hat.tolist(),
                'q_hat_raw': (self.norm_params.sigma * self.q_hat).tolist(),
                'target_coverage': 1.0 - self.alpha,
                'local_residual_std': self._local_residual_std,
            }
        }

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

        for i in range(len(X_test)):
            history_raw = self.norm_params.denormalize(X_test[i].flatten())

            t0 = time.time()
            pred = self.predict_with_uncertainty(history_raw)
            times_ms.append((time.time() - t0) * 1000)

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
            'method': 'scp',
            'empirical_coverage': float(np.mean(coverages)),
            'target_coverage': 1.0 - self.alpha,
            'mean_interval_width': float(np.mean(widths)),
            'mean_mape': float(np.mean(mapes)),
            'mean_inference_ms': float(np.mean(times_ms)),
            'tier_distribution': tiers,
            'n_test_samples': len(X_test),
            'q_hat': self.q_hat.tolist(),
        }

    def save(self, output_dir: str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.base_model:
            torch.save(self.base_model.state_dict(), output_dir / 'base_model.pt')

        config = {
            'method': 'scp',
            'alpha': self.alpha,
            'h': self.h,
            'k': self.k,
            'q_hat': self.q_hat.tolist() if self.q_hat is not None else None,
            'local_residual_std': self._local_residual_std,
        }
        if self.norm_params:
            config['normalization'] = {
                'mu': self.norm_params.mu,
                'sigma': self.norm_params.sigma,
            }
        with open(output_dir / 'scp_config.yaml', 'w') as f:
            yaml.dump(config, f)

        logger.info("SCP saved to %s", output_dir)

    @classmethod
    def load(cls, model_dir: str, device: str = 'cpu') -> 'SplitConformal':
        model_dir = Path(model_dir)
        with open(model_dir / 'scp_config.yaml') as f:
            config = yaml.safe_load(f)

        scp = cls(
            alpha=config['alpha'],
            h=config['h'],
            k=config['k'],
            device=device,
        )
        scp.q_hat = np.array(config['q_hat'], dtype=np.float32)
        scp._local_residual_std = config['local_residual_std']

        if 'normalization' in config:
            scp.norm_params = NormalizationParams(
                mu=config['normalization']['mu'],
                sigma=config['normalization']['sigma'],
            )

        scp.base_model = WorkloadGRU(
            input_size=1, hidden_size=64, num_layers=2,
            output_size=scp.k, dropout=0.2,
        ).to(device)
        scp.base_model.load_state_dict(
            torch.load(model_dir / 'base_model.pt',
                      weights_only=True, map_location=device)
        )
        scp.base_model.eval()

        logger.info("SCP loaded from %s", model_dir)
        return scp
