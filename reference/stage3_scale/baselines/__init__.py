"""Baseline autoscaling methods for P3 experiments.

Provides three controller-based baselines (predictive, predictive-safety,
BASE-inspired) and a KEDA ScaledObject manifest. All baselines implement
the same controller subprocess pattern as the UQ methods, allowing the
orchestrator to treat all 8 methods uniformly.

Controller entry point:
    baselines/controller.py --mode {predictive|predictive-safety|base-inspired}
"""

__all__ = [
    "CONTROLLER_SCRIPT",
    "KEDA_SCALEDOBJECT_YAML",
]

from pathlib import Path

CONTROLLER_SCRIPT = Path(__file__).resolve().parent / 'controller.py'
KEDA_SCALEDOBJECT_YAML = Path(__file__).resolve().parent / 'keda-scaledobject.yaml'
