"""Uncertainty estimation interface and confidence tiers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import torch


class UncertaintyQuantifier(ABC):
    """Abstract interface for UQ methods."""

    method: str              # 'be', 'scp', 'qr'
    alpha: float = 0.1       # Confidence level (intervals at 1-alpha)
    h: int = 60              # History window
    k: int = 2               # Forecast horizon
    norm_params: Optional[object] = None  # NormalizationParams for denormalization
    device: str = 'cpu'

    @abstractmethod
    def fit(self, train_data: tuple, calibration_data: tuple = None) -> 'UncertaintyQuantifier':
        """Train on normalized (X, y): X has shape (N, h) or (N, h, 1), y has shape (N, k)."""

    @abstractmethod
    def predict_with_uncertainty(self, history: np.ndarray) -> dict:
        """Accept raw RPS history of shape (h,). Return forecast and bounds of shape (k,),
        confidence_score (larger means less certain), tier, method and metadata.
        """

    @abstractmethod
    def evaluate_coverage(self, test_data: tuple) -> dict:
        """Evaluate empirical coverage, interval width, and calibration."""

    def save(self, output_dir: str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, model_dir: str) -> 'UncertaintyQuantifier':
        raise NotImplementedError


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
