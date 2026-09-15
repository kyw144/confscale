"""Baseline autoscaling methods."""

__all__ = [
    "CONTROLLER_SCRIPT",
    "KEDA_SCALEDOBJECT_YAML",
]

from pathlib import Path

CONTROLLER_SCRIPT = Path(__file__).resolve().parent / 'controller.py'
KEDA_SCALEDOBJECT_YAML = Path(__file__).resolve().parent / 'keda-scaledobject.yaml'
