"""
Uncertainty Quantification Modules — P3 Research Objective 05.

Three complementary methods wrapping the GRU predictor to produce
prediction intervals [r̂_lo, r̂_hi] and confidence scores.

    BE  — Bootstrap Ensemble    (train B GRUs on bootstrap samples)
    SCP — Split-Conformal       (distribution-free finite-sample coverage)
    QR  — Quantile Regression   (direct quantile prediction via pinball loss)

All three implement the UncertaintyQuantifier interface consumed by
the ConfidenceScaler operator (task 07) and the experiment orchestrator.

Usage:
    from uq import BootstrapEnsemble, SplitConformal, QuantileRegressor
    from uq.evaluate import evaluate_uq_method, compare_methods
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import torch


class UncertaintyQuantifier(ABC):
    """Abstract interface for UQ methods.

    Contract consumed by:
      - ConfidenceScaler operator (task 07)
      - Experiment orchestrator methods (confscale-*)  
      - Analysis pipeline (task 08)

    Each implementation stores its state (models, calibration params)
    and exposes predict_with_uncertainty() for online inference.
    """

    method: str              # 'be', 'scp', 'qr'
    alpha: float = 0.1       # Confidence level (intervals at 1-alpha)
    h: int = 60              # History window
    k: int = 2               # Forecast horizon
    norm_params: Optional[object] = None  # NormalizationParams for denormalization
    device: str = 'cpu'

    @abstractmethod
    def fit(self, train_data: tuple, calibration_data: tuple = None) -> 'UncertaintyQuantifier':
        """Train and/or calibrate the UQ method.

        Args:
            train_data: (X_train, y_train) — numpy arrays, normalized
                        X_train shape (N, h) or (N, h, 1)
                        y_train shape (N, k)
            calibration_data: (X_cal, y_cal) for SCP; optional for BE/QR
        """

    @abstractmethod
    def predict_with_uncertainty(self, history: np.ndarray) -> dict:
        """Produce point forecast + uncertainty bounds for a single history window.

        Args:
            history: shape (h,) — last h request rate observations (raw RPS)

        Returns:
            dict with keys:
              - point_forecast:   np.ndarray (k,) — mean/median prediction
              - ci_lower:         np.ndarray (k,) — lower bound (≥0 guaranteed)
              - ci_upper:         np.ndarray (k,) — upper bound
              - confidence_score: float — [0, 1] where higher = more uncertain
              - tier:             int — 1 (high), 2 (medium), 3 (low)
              - method:           str — 'be', 'scp', or 'qr'
              - metadata:         dict — method-specific diagnostics
        """

    @abstractmethod
    def evaluate_coverage(self, test_data: tuple) -> dict:
        """Evaluate empirical coverage, interval width, and calibration."""

    def save(self, output_dir: str) -> None:
        """Save models and calibration state to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, model_dir: str) -> 'UncertaintyQuantifier':
        """Load a saved UQ method from disk."""
        raise NotImplementedError


# ── Confidence Tier Mapping ───────────────────────────────────────────

TIER_1_CV = 0.1    # CV < 0.1 → Tier 1 (high confidence)
TIER_2_CV = 0.3    # CV < 0.3 → Tier 2 (medium confidence)
                    # CV ≥ 0.3 → Tier 3 (low confidence)

# SCP-specific thresholds (normalized interval width)
TIER_1_WIDTH = 0.2
TIER_2_WIDTH = 0.6


def cv_to_tier(cv: float) -> int:
    """Map coefficient of variation to confidence tier."""
    if cv < TIER_1_CV:
        return 1
    elif cv < TIER_2_CV:
        return 2
    else:
        return 3


def width_to_tier(normalized_width: float) -> int:
    """Map normalized interval width to confidence tier (SCP)."""
    if normalized_width < TIER_1_WIDTH:
        return 1
    elif normalized_width < TIER_2_WIDTH:
        return 2
    else:
        return 3
