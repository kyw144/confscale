"""Bootstrap Ensemble (BE) — Uncertainty Quantification via model diversity."""

import sys
from pathlib import Path
import copy
import time
import logging

import numpy as np
import torch
import yaml

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from . import UncertaintyQuantifier, cv_to_tier
from predictor.gru_model import WorkloadGRU
from predictor.data import NormalizationParams

logger = logging.getLogger(__name__)


class BootstrapEnsemble(UncertaintyQuantifier):
    """B bootstrap GRU models; intervals from empirical quantiles."""

    method = 'be'

    def __init__(self, B: int = 10, alpha: float = 0.1,
                 h: int = 60, k: int = 2, device: str = 'cpu',
                 seed: int = 42):
        self.B = B
        self.alpha = alpha
        self.h = h
        self.k = k
        self.device = device
        self.seed = seed

        self.models: list[WorkloadGRU] = []
        self.norm_params: Optional[NormalizationParams] = None

        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.2
        self.epochs = 200
        self.lr = 1e-3
        self.batch_size = 64
        self.patience = 20

        self._rng = np.random.RandomState(seed)

    def fit(self,
            train_data: tuple,
            calibration_data: tuple = None) -> 'BootstrapEnsemble':
        """Train B GRU models on independent bootstrap samples."""
        X_train, y_train = train_data
        
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        if X_train.ndim == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # 80/20 train/cal split from train_data for later evaluation
        n = len(X_train)
        n_train = int(0.8 * n)
        self._X_cal = X_train[n_train:]
        self._y_cal = y_train[n_train:]
        X_train_be = X_train[:n_train]
        y_train_be = y_train[:n_train]

        N = len(X_train_be)
        logger.info("BE: Training %d ensemble members on %d bootstrap samples each",
                     self.B, N)

        for b in range(self.B):
            t0 = time.time()

            indices = self._rng.choice(N, size=N, replace=True)
            X_boot = X_train_be[indices]
            y_boot = y_train_be[indices]

            model = WorkloadGRU(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=self.k,
                dropout=self.dropout,
            ).to(self.device)

            model = _train_gru(
                model=model,
                X=X_boot,
                y=y_boot,
                epochs=self.epochs,
                lr=self.lr,
                batch_size=self.batch_size,
                patience=self.patience,
                device=self.device,
                seed=self.seed + b,
            )

            self.models.append(model)
            dt = time.time() - t0
            logger.info("  Member %d/%d trained in %.1fs", b + 1, self.B, dt)

        logger.info("BE: Ensemble of %d models trained", self.B)
        return self

    def predict_with_uncertainty(self, history: np.ndarray) -> dict:
        """Predict with uncertainty from ensemble spread."""
        if not self.models:
            raise RuntimeError("BE not fitted. Call fit() first.")
        if self.norm_params is None:
            raise RuntimeError("BE missing norm_params. Set before calling predict.")

        if len(history) != self.h:
            raise ValueError(f"History must have {self.h} values, got {len(history)}")

        # Normalize history
        x = self.norm_params.normalize(history.astype(np.float32, copy=False))
        x_tensor = torch.from_numpy(x).to(self.device).reshape(1, self.h, 1)

        all_preds_norm = []
        with torch.inference_mode():
            for model in self.models:
                model.eval()
                pred_norm = model(x_tensor).cpu().numpy().flatten()
                all_preds_norm.append(pred_norm)

        preds_norm = np.array(all_preds_norm)  # (B, k)

        preds_rps = np.array([
            self.norm_params.denormalize(preds_norm[i])
            for i in range(self.B)
        ])

        point_forecast = preds_rps.mean(axis=0)

        ci_lower = np.percentile(preds_rps, self.alpha / 2 * 100, axis=0)
        ci_upper = np.percentile(preds_rps, (1 - self.alpha / 2) * 100, axis=0)

        np.maximum(ci_lower, 0.0, out=ci_lower)
        np.maximum(ci_upper, 0.0, out=ci_upper)
        np.maximum(point_forecast, 0.0, out=point_forecast)

        ensemble_std = preds_rps.std(axis=0)
        ensemble_mean = preds_rps.mean(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_per_step = np.where(ensemble_mean > 1e-6,
                                   ensemble_std / ensemble_mean, 0.0)
        confidence_score = float(np.mean(cv_per_step))

        tier = cv_to_tier(confidence_score)

        return {
            'point_forecast': point_forecast,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_score': confidence_score,
            'tier': tier,
            'method': 'be',
            'metadata': {
                'ensemble_std': ensemble_std.tolist(),
                'ensemble_mean': ensemble_mean.tolist(),
                'cv_per_step': cv_per_step.tolist(),
                'B': self.B,
                'alpha': self.alpha,
            }
        }

    def evaluate_coverage(self, test_data: tuple) -> dict:
        """Evaluate empirical coverage on held-out test data."""
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
            # Need to denormalize history before calling predict_with_uncertainty
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
            'method': 'be',
            'empirical_coverage': float(np.mean(coverages)),
            'target_coverage': 1.0 - self.alpha,
            'mean_interval_width': float(np.mean(widths)),
            'mean_mape': float(np.mean(mapes)),
            'mean_inference_ms': float(np.mean(times_ms)),
            'tier_distribution': tiers,
            'n_test_samples': len(X_test),
            'ensemble_size': self.B,
        }

    def save(self, output_dir: str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for b, model in enumerate(self.models):
            torch.save(model.state_dict(), output_dir / f"member_{b:02d}.pt")

        config = {
            'method': 'be',
            'B': self.B,
            'alpha': self.alpha,
            'h': self.h,
            'k': self.k,
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
        with open(output_dir / 'be_config.yaml', 'w') as f:
            yaml.dump(config, f)

        logger.info("BE saved to %s (%d members)", output_dir, self.B)

    @classmethod
    def load(cls, model_dir: str, device: str = 'cpu') -> 'BootstrapEnsemble':
        model_dir = Path(model_dir)
        with open(model_dir / 'be_config.yaml') as f:
            config = yaml.safe_load(f)

        be = cls(
            B=config['B'],
            alpha=config['alpha'],
            h=config['h'],
            k=config['k'],
            device=device,
            seed=config['seed'],
        )
        be.hidden_size = config['hidden_size']
        be.num_layers = config['num_layers']
        be.dropout = config['dropout']

        # Load normalization
        if 'normalization' in config:
            be.norm_params = NormalizationParams(
                mu=config['normalization']['mu'],
                sigma=config['normalization']['sigma'],
            )

        for b in range(config['B']):
            model = WorkloadGRU(
                input_size=1,
                hidden_size=be.hidden_size,
                num_layers=be.num_layers,
                output_size=be.k,
                dropout=be.dropout,
            ).to(device)
            model.load_state_dict(
                torch.load(model_dir / f"member_{b:02d}.pt",
                          weights_only=True, map_location=device)
            )
            model.eval()
            be.models.append(model)

        logger.info("BE loaded from %s (%d members)", model_dir, config['B'])
        return be


def _train_gru(model: WorkloadGRU, X: np.ndarray, y: np.ndarray,
               epochs: int, lr: float, batch_size: int, patience: int,
               device: str, seed: int) -> WorkloadGRU:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(X)
    n_train = int(0.8 * n)

    X_tr = torch.from_numpy(X[:n_train]).to(device)
    y_tr = torch.from_numpy(y[:n_train]).to(device)
    X_val = torch.from_numpy(X[n_train:]).to(device)
    y_val = torch.from_numpy(y[n_train:]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr[idx]
            yb = y_tr[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.inference_mode():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model
