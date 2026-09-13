"""
Adaptive Conformal Inference (ACI) baseline — proportional-only α tracker.

Implements Gibbs & Candès (NeurIPS 2021), "Adaptive Conformal Inference Under
Distribution Shift" (arXiv:2106.00170):

    α_{t+1} = α_t + η · (α_target − 1{y_t ∉ [L_t, U_t]})

This is the principled-but-simpler comparator to :class:`ConformalPID` and a
special case of it with K_I = K_D = 0. Implemented as a thin subclass so the
equivalence is true by construction (and any future PID fix automatically
benefits ACI). Exposed as a distinct class so the four-recalibrator
experimental axis (static / rolling-origin / ACI / PID) reads cleanly in the
method registry.

Sign convention matches :class:`ConformalPID` — see that module for the
derivation.
"""

from __future__ import annotations

from typing import Optional

from .conformal_pid import ConformalPID


class ACI(ConformalPID):
    """Proportional-only recalibrator — Gibbs-Candès ACI.

    Args:
        target_alpha: Target miscoverage level (e.g. 0.1 for 90% intervals).
        eta: Step size (proportional gain). Aliased to ``k_p`` on the
            underlying :class:`ConformalPID`.
        alpha_init: Initial working α. Falls back to ``target_alpha`` when None.
        residual_buffer_size: Trailing residual buffer length (FIFO).
        alpha_clip: ``(low, high)`` clamp on α.
    """

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
        """Same as parent, plus the ``eta`` alias for clarity in logs."""
        s = super().state()
        s["eta"] = float(self.eta)
        return s
