"""Adaptive Conformal Inference: PID with zero integral and derivative gains.

Gibbs & Candès (2021), arXiv:2106.00170.
"""

from __future__ import annotations

from typing import Optional

from .conformal_pid import ConformalPID


class ACI(ConformalPID):
    """Conformal PID with proportional gain eta and zero integral/derivative gains."""

    def __init__(
        self,
        target_alpha: float = 0.1,
        eta: float = 0.1,
        alpha_init: Optional[float] = None,
        residual_buffer_size: int = 200,
        alpha_clip: tuple[float, float] = (1e-4, 0.5),
    ) -> None:
        super().__init__(
            target_alpha=target_alpha,
            k_p=eta,
            k_i=0.0,
            k_d=0.0,
            alpha_init=alpha_init,
            residual_buffer_size=residual_buffer_size,
            alpha_clip=alpha_clip,
        )
        self.eta = float(eta)

    def state(self) -> dict:
        s = super().state()
        s["eta"] = float(self.eta)
        return s
