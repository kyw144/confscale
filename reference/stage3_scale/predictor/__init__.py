"""
GRU Predictor — the public API consumed by downstream components.

Loads a trained model + config, provides predict() and predict_with_history().
This is the contract used by UQ modules (task 05), ConfidenceScaler (task 07),
and the analysis pipeline (task 08).
"""

from pathlib import Path
import numpy as np
import torch
import yaml
from typing import Optional

from .gru_model import WorkloadGRU
from .data import NormalizationParams


class GRUPredictor:
    """Load a trained GRU model and make predictions.

    Usage:
        predictor = GRUPredictor('models/gru_compute-worker_diurnal/')
        result = predictor.predict(history_array)  # shape (h,)
        result = predictor.predict_with_history(timestamps, values)
    """

    def __init__(self, model_dir: str, device: Optional[str] = None):
        """
        Args:
            model_dir: Path to directory containing model.pt + gru_config.yaml
            device: 'cpu', 'mps', or None (auto-detect)
        """
        model_dir = Path(model_dir)
        self.model_dir = model_dir

        # Load config
        with open(model_dir / 'gru_config.yaml') as f:
            self.config = yaml.safe_load(f)

        preproc = self.config['preprocessing']
        self.h = preproc['h']
        self.k = preproc['k']
        self.norm = NormalizationParams(
            mu=preproc['normalization']['mu'],
            sigma=preproc['normalization']['sigma'],
        )

        # Load model
        model_cfg = self.config['model']
        self.model = WorkloadGRU(
            input_size=model_cfg['input_size'],
            hidden_size=model_cfg['hidden_size'],
            num_layers=model_cfg['num_layers'],
            output_size=model_cfg['output_size'],
            dropout=model_cfg['dropout'],
        )

        # Device selection — CPU is faster for small models (<5ms forward pass)
        if device is None:
            device = 'cpu'
        self.device = device
        self.model.load_state_dict(
            torch.load(model_dir / 'model.pt', weights_only=True,
                      map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, history: np.ndarray) -> dict:
        """Predict future request rates from the last h observations.

        Args:
            history: shape (h,) — last h request rate observations (raw RPS)

        Returns:
            {
                'point_forecast': np.ndarray shape (k,) — predicted rates in RPS,
                'forecast_horizon_s': [30, 60, ...] — timestamps for each step
            }
        """
        if len(history) != self.h:
            raise ValueError(
                f"History must have exactly {self.h} values, got {len(history)}"
            )

        # Normalize and convert to tensor efficiently
        x = self.norm.normalize(history.astype(np.float32, copy=False))
        x_tensor = torch.from_numpy(x).to(self.device).reshape(1, self.h, 1)

        with torch.inference_mode():
            pred_norm = self.model(x_tensor).cpu().numpy().flatten()

        # Denormalize and floor at 0 (non-negativity enforced here, not in model)
        pred_rps = self.norm.denormalize(pred_norm)
        np.maximum(pred_rps, 0.0, out=pred_rps)

        return {
            'point_forecast': pred_rps,
            'forecast_horizon_s': [(i + 1) * 30 for i in range(self.k)],
        }

    def predict_with_history(
        self,
        timestamps: list[float],
        values: list[float],
    ) -> dict:
        """Convenience: extract last h values from a timeseries, run predict().

        Args:
            timestamps: List of timestamps (seconds)
            values: List of RPS values

        Returns:
            Same as predict()
        """
        if len(values) < self.h:
            raise ValueError(
                f"Need at least {self.h} observations, got {len(values)}"
            )
        return self.predict(np.array(values[-self.h:], dtype=np.float32))
