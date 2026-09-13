"""Experiment Orchestrator — runs the P3 experimental matrix.

Provides:
    run_matrix() — execute (method × workload × replicate) matrix
    MethodConfig / METHOD_REGISTRY — scaling method abstraction
"""

from .run_matrix import run_matrix
from .methods import MethodConfig, METHOD_REGISTRY, get_method, list_methods

__all__ = [
    "run_matrix",
    "MethodConfig",
    "METHOD_REGISTRY",
    "get_method",
    "list_methods",
]
