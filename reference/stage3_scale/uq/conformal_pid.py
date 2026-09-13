"""
Conformal PID recalibrator — online α-tracking for streaming SCP.

Implements the controller in Angelopoulos, Candès & Tibshirani (NeurIPS 2023),
"Conformal PID Control for Time-Series Prediction" (arXiv:2307.16895). For each
prediction cycle we observe whether the realized value fell inside the current
prediction interval, then nudge the working miscoverage level α via a
proportional-integral-derivative law. The next half-width is the
(1 − α)-quantile of the trailing residual buffer.

Sign convention. The proportional, integral and derivative terms move α *down*
on a miss (so the next interval *widens*) and *up* on a cover, matching the
Gibbs-Candès ACI update `α_{t+1} = α_t + γ(α_target − err_t)`. The signed error
``e_t`` is therefore ``α_target − 1{miss}`` — positive when we are over-covering
relative to target. With positive K-gains this drives empirical coverage
towards the target; a positive e_t (over-cover) raises α (tightens the interval)
and a negative e_t (miss) lowers α (widens the interval).

The class is unit-agnostic: residuals are stored in whatever space the caller
chose, and ``quantile()`` returns a half-width in that same space. The
controller passes normalized residuals (matching the existing
``confscale-scp-online`` rolling-origin path) and converts back to raw RPS via
the predictor's normalization σ before feeding the scaler.

Single-stream (h=0) for v1, matching the ``CoverageMonitor`` Analyze surface.
Multi-horizon extension is future work.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import math


class EmptyResidualBufferError(RuntimeError):
    """Raised when ``quantile()`` is called before any residual has been recorded."""


class ConformalPID:
    """Online miscoverage-tracking conformal recalibrator.

    Args:
        target_alpha: Target miscoverage level (e.g. 0.1 for 90% intervals).
        k_p: Proportional gain.
        k_i: Integral gain.
        k_d: Derivative gain.
        alpha_init: Initial working α. Falls back to ``target_alpha`` when None.
        residual_buffer_size: Trailing residual buffer length (FIFO).
        alpha_clip: ``(low, high)`` clamp on α; updates outside the range are
            clipped to the boundary.

    Defaults follow the E1 brief (NeurIPS 2023 §4 forecasting magnitudes).
    """

    def __init__(
        self,
        target_alpha: float = 0.1,
        k_p: float = 0.1,
        k_i: float = 0.01,
        k_d: float = 0.05,
        alpha_init: Optional[float] = None,
        residual_buffer_size: int = 200,
        alpha_clip: tuple[float, float] = (1e-4, 0.5),
    ) -> None:
        if not (0.0 < target_alpha < 1.0):
            raise ValueError(
                f"target_alpha must be in (0, 1), got {target_alpha}"
            )
        if residual_buffer_size <= 0:
            raise ValueError(
                f"residual_buffer_size must be positive, got {residual_buffer_size}"
            )
        low, high = alpha_clip
        if not (0.0 < low < high < 1.0):
            raise ValueError(
                f"alpha_clip must satisfy 0 < low < high < 1, got {alpha_clip}"
            )

        self.target_alpha = float(target_alpha)
        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)
        self.alpha_clip = (float(low), float(high))

        initial = float(alpha_init) if alpha_init is not None else self.target_alpha
        self.alpha = self._clip(initial)

        self.residuals: deque[float] = deque(maxlen=residual_buffer_size)
        self.integral: float = 0.0
        self.last_error: float = 0.0
        self.last_derivative: float = 0.0
        self.steps: int = 0

    # ── Update ──────────────────────────────────────────────────────────

    def update(self, residual: float, miscovered: bool) -> None:
        """Run one PID cycle.

        Appends ``|residual|`` to the trailing buffer, then advances α using
        the signed coverage error ``e_t = α_target − 1{miscovered}``.
        """
        self.residuals.append(abs(float(residual)))

        miss = 1.0 if miscovered else 0.0
        e_t = self.target_alpha - miss
        self.integral += e_t
        derivative = e_t - self.last_error

        new_alpha = (
            self.alpha
            + self.k_p * e_t
            + self.k_i * self.integral
            + self.k_d * derivative
        )
        self.alpha = self._clip(new_alpha)
        self.last_error = e_t
        self.last_derivative = derivative
        self.steps += 1

    # ── Output ──────────────────────────────────────────────────────────

    def quantile(self) -> float:
        """Conformal half-width at the current adjusted α.

        Returns the ``ceil((1 − α)·(n + 1))``-th smallest absolute residual
        from the buffer (standard split-conformal finite-sample correction).
        Raises :class:`EmptyResidualBufferError` if the buffer is empty —
        the caller is expected to fall back to the offline-trained SCP
        quantile in that case (see E1 brief §1 edge cases).
        """
        n = len(self.residuals)
        if n == 0:
            raise EmptyResidualBufferError(
                "ConformalPID.quantile() called with empty residual buffer; "
                "caller must fall back to the offline SCP quantile."
            )
        sorted_residuals = sorted(self.residuals)
        q_index = math.ceil((1.0 - self.alpha) * (n + 1)) - 1
        q_index = max(0, min(q_index, n - 1))
        return sorted_residuals[q_index]

    # ── Introspection ───────────────────────────────────────────────────

    def state(self) -> dict:
        """Snapshot of internal state for Prometheus + tests."""
        return {
            "alpha": float(self.alpha),
            "target_alpha": float(self.target_alpha),
            "integral": float(self.integral),
            "derivative": float(self.last_derivative),
            "last_error": float(self.last_error),
            "steps": int(self.steps),
            "buffer_size": int(len(self.residuals)),
            "k_p": float(self.k_p),
            "k_i": float(self.k_i),
            "k_d": float(self.k_d),
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    def _clip(self, value: float) -> float:
        low, high = self.alpha_clip
        if value < low:
            return low
        if value > high:
            return high
        return float(value)
