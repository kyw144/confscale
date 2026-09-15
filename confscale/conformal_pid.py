"""Online conformal PID recalibration.

Angelopoulos, Candès & Tibshirani (2023), arXiv:2307.16895.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import math


class EmptyResidualBufferError(RuntimeError):
    """Raised when ``quantile()`` is called before any residual has been recorded."""


class ConformalPID:
    """Track miscoverage using absolute residuals in caller-supplied units."""

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

    def update(self, residual: float, miscovered: bool) -> None:
        """The signed error is target_alpha - miss; proportional feedback widens on a miss."""
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

    def quantile(self) -> float:
        """Return the residual at the clipped ceil((1-alpha)*(n+1)) rank.

        Raises EmptyResidualBufferError before the first residual; callers may use the offline quantile.
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

    def state(self) -> dict:
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

    def _clip(self, value: float) -> float:
        low, high = self.alpha_clip
        if value < low:
            return low
        if value > high:
            return high
        return float(value)
