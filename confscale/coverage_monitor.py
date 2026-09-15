#!/usr/bin/env python3
"""FIFO coverage monitoring for one-step prediction intervals."""
from __future__ import annotations

import time
from collections import deque
from typing import Optional


class CoverageMonitor:
    """Score the oldest pending interval after its observation arrives."""

    # Warning-band lower bound expressed as a fraction of target_coverage:
    # below this, alert tips from warning (1) to critical (2).
    CRITICAL_FRACTION_OF_TARGET = 0.85

    def __init__(self, window_size: int = 30, target_coverage: float = 0.90):
        if window_size <= 0:
            raise ValueError(
                f"CoverageMonitor window_size must be positive, got {window_size}"
            )
        if not (0.0 < target_coverage <= 1.0):
            raise ValueError(
                f"CoverageMonitor target_coverage must be in (0, 1], "
                f"got {target_coverage}"
            )
        self.window_size = window_size
        self.target_coverage = float(target_coverage)
        self.pending: list[tuple[float, float, float]] = []
        self.validated: deque[bool] = deque(maxlen=window_size)
        # Start nominal — empty window means no failures observed yet.
        self.trailing_coverage: float = 1.0
        self.alert_state: int = 0
        self.lifetime_validated: int = 0
        self.lifetime_covered: int = 0

    def record_prediction(self, ci_lower_h0: float, ci_upper_h0: float) -> None:
        """Stash an h=0 prediction interval for next-iteration validation."""
        self.pending.append((float(ci_lower_h0), float(ci_upper_h0), time.time()))

    def validate_pending(self, observed_rps: float) -> Optional[bool]:
        """Return coverage of the oldest pending interval, or None when none is pending."""
        if not self.pending:
            return None
        ci_lower, ci_upper, _recorded_at = self.pending.pop(0)
        covered = bool(ci_lower <= float(observed_rps) <= ci_upper)
        self.validated.append(covered)
        self.lifetime_validated += 1
        if covered:
            self.lifetime_covered += 1
        self._recompute_state()
        return covered

    def _recompute_state(self) -> None:
        if not self.validated:
            self.trailing_coverage = 1.0
        else:
            self.trailing_coverage = (
                sum(1 for v in self.validated if v) / len(self.validated)
            )
        critical_threshold = self.CRITICAL_FRACTION_OF_TARGET * self.target_coverage
        if self.trailing_coverage >= self.target_coverage:
            self.alert_state = 0
        elif self.trailing_coverage >= critical_threshold:
            self.alert_state = 1
        else:
            self.alert_state = 2

    def coverage_shortfall(self) -> float:
        """How far below target_coverage trailing coverage sits (>= 0)."""
        return max(0.0, self.target_coverage - self.trailing_coverage)

    def needs_recalibration(self) -> bool:
        """Return True in the critical alert band; no recalibration is executed."""
        return self.alert_state >= 2

    def get_state(self) -> dict:
        return {
            'trailing_coverage': round(float(self.trailing_coverage), 4),
            'coverage_shortfall': round(float(self.coverage_shortfall()), 4),
            'alert_state': int(self.alert_state),
            'target_coverage': float(self.target_coverage),
            'window_size': int(self.window_size),
            'pending': len(self.pending),
            'validated': len(self.validated),
            'covered': sum(1 for v in self.validated if v),
        }
