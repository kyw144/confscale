#!/usr/bin/env python3
"""
CoverageMonitor — online trailing empirical-coverage signal.

Used by both the orchestrator's confidence-aware controller
(``orchestrator/controller.py``) and the baseline controller
(``baselines/controller.py``) when running coverage-monitored methods.
The class is pure measurement — no Plan-side action, no recalibration —
so it slots into either controller's main loop with the same wiring:

  1. After observing this iteration's RPS, call ``validate_pending(rps)``
     to score the prior iteration's h=0 CI.
  2. After producing this iteration's CI, call
     ``record_prediction(ci_lower, ci_upper)`` to stash it for next-
     iteration validation.
  3. Read ``trailing_coverage`` / ``alert_state`` / ``get_state()`` for
     logs and metrics.

Extracted from ``orchestrator/controller.py`` so the baseline path can
import it without pulling in the orchestrator's controller-only deps
(Prometheus client, tiered hysteresis, recalibrators).
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional


class CoverageMonitor:
    """Trailing empirical-coverage monitor for the predictive controller.

    Each iteration that produces a forecast pushes its h=0 prediction
    interval ``(ci_lower, ci_upper)`` to ``pending`` via
    ``record_prediction``. The next iteration validates the oldest
    pending entry against the realized RPS by calling
    ``validate_pending`` — the matched residual is appended to a
    fixed-size FIFO window of in-interval booleans.

    ``trailing_coverage`` is the in-interval fraction over the current
    window. ``alert_state`` is a coarse band on that fraction relative
    to ``target_coverage`` T:

      - 0 (nominal):  trailing_coverage >= T
      - 1 (warning):  0.85 * T <= trailing_coverage < T
      - 2 (critical): trailing_coverage < 0.85 * T

    This class only *measures* — it is the Analyze side of the
    coverage-monitored controller (Paper 3 reframe brief, C2-Analyze).
    The Plan-side action (recalibrate / escalate when shortfall fires)
    is C2-Plan / C4 and waits on the E1 recalibrator-choice decision;
    ``needs_recalibration`` is provided as a stub for that wiring.
    """

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
        # Lifetime counters for the operator_metrics_summary.json section.
        self.lifetime_validated: int = 0
        self.lifetime_covered: int = 0

    def record_prediction(self, ci_lower_h0: float, ci_upper_h0: float) -> None:
        """Stash an h=0 prediction interval for next-iteration validation."""
        self.pending.append((float(ci_lower_h0), float(ci_upper_h0), time.time()))

    def validate_pending(self, observed_rps: float) -> Optional[bool]:
        """Score the oldest pending interval against observed_rps.

        Returns True if covered, False if not, or None if there was no
        pending entry (e.g., the first iteration after start).
        """
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
        """Stub for the Plan side (C2-Plan / C4, E1-blocked).

        Default reading: True when the alert band is critical. The
        Plan-side recalibrator will pick its own threshold once E1 is
        resolved; this is just a conservative placeholder.
        """
        return self.alert_state >= 2

    def get_state(self) -> dict:
        """Snapshot for inclusion in controller_scale_log.json entries."""
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
