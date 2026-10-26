"""Replica targets and tier hysteresis."""
import time
import statistics
from collections import deque
from typing import Optional
import numpy as np

TIER_HIGH_CONF = 0.15

TIER_MED_CONF = 0.4

TIER_DOWNGRADE_DELAY = 5

SCALE_UP_COOLDOWN_S = 30

SCALE_DOWN_COOLDOWN_T1_S = 60

SCALE_DOWN_COOLDOWN_T2_S = 90

SCALE_DOWN_COOLDOWN_T3_S = 120

def _tier_cooldown(tier: int) -> float:
    if tier == 3:
        return SCALE_DOWN_COOLDOWN_T3_S
    elif tier == 2:
        return SCALE_DOWN_COOLDOWN_T2_S
    return SCALE_DOWN_COOLDOWN_T1_S

def compute_target_replicas(
    point_forecast: np.ndarray,
    confidence_score: float,
    ci_upper: np.ndarray = None,
    slo_capacity: float = 10.0,
    target_util: float = 0.7,
    safety_factor: float = 2.0,
    min_replicas: int = 1,
    max_replicas: int = 20,
    policy: str = "tier",
    lambda_risk: Optional[float] = None,
    h0_only: bool = False,
    point_bias: Optional[float] = None,
) -> tuple[int, int]:
    """Return (tier, replicas) from the maximum horizon demand.

    lambda_risk overrides policy; ci-upper returns tier 0 while hysteresis tracks confidence tiers.
    h0_only uses the corrected first forecast; raw confidence still selects cooldowns.
    """
    # Always classify tier so hysteresis cooldown selection works.
    if confidence_score < TIER_HIGH_CONF:
        classified_tier = 1
    elif confidence_score < TIER_MED_CONF:
        classified_tier = 2
    else:
        classified_tier = 3

    if h0_only or point_bias is not None:
        try:
            h0 = float(point_forecast[0])
            bias = 0.0 if point_bias is None else float(point_bias)
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError('Corrected point control requires finite h0 and bias') from error
        if not np.isfinite(h0) or not np.isfinite(bias):
            raise ValueError('Corrected point control requires finite h0 and bias')
        effective_rate = max(0.0, h0 - bias)
        tier = classified_tier
    elif lambda_risk is not None:
        upper = ci_upper if ci_upper is not None else point_forecast * 1.5
        upper_gap = np.maximum(upper - point_forecast, 0.0)
        effective = point_forecast + lambda_risk * upper_gap
        effective_rate = float(np.max(effective))
        tier = classified_tier
    elif policy == "ci-upper":
        if ci_upper is not None:
            effective_rate = float(np.max(ci_upper))
        else:
            effective_rate = float(np.max(point_forecast * 1.5))
        tier = 0  # sentinel: no tier policy active
    else:
        # Default 3-tier policy
        tier = classified_tier
        if tier == 1:
            effective_rate = float(np.max(point_forecast))
        elif tier == 2:
            if ci_upper is not None:
                std_est = (ci_upper - point_forecast) / 1.645  # 90% CI: z=1.645
            else:
                std_est = point_forecast * confidence_score
            effective_rate = float(np.max(point_forecast + safety_factor * std_est))
        else:
            if ci_upper is not None:
                effective_rate = float(np.max(ci_upper))
            else:
                effective_rate = float(np.max(point_forecast * 1.5))

    replicas = int(np.ceil(effective_rate / (slo_capacity * target_util)))
    replicas = max(min_replicas, min(replicas, max_replicas))

    return tier, replicas


class PointBiasCorrector:
    """Match the immediately preceding h0 forecast to the current observation.

    Missing forecasts or observations break the pair. Unmatched forecasts are
    not retained across steps; matched errors remain in the bounded FIFO.
    """

    def __init__(self, window: int):
        if isinstance(window, bool) or not isinstance(window, int) or not 0 <= window <= 10:
            raise ValueError('point bias window must be an integer from 0 to 10')
        self.residuals = deque(maxlen=window)
        self.previous_h0 = None

    def step(self, rps, pred: Optional[dict]) -> dict:
        if self.previous_h0 is not None and rps is not None and np.isfinite(rps):
            self.residuals.append(self.previous_h0 - float(rps))
        point = pred.get('point_forecast') if pred is not None else None
        h0 = float(point[0]) if point is not None and len(point) else None
        if h0 is not None and not np.isfinite(h0):
            h0 = None
        self.previous_h0 = h0
        bias = statistics.mean(self.residuals) if self.residuals else 0.0
        return {'raw_h0': h0, 'bias_rps': bias, 'matched_errors': len(self.residuals),
                'corrected_rps': max(0.0, h0 - bias) if h0 is not None else None}


class HysteresisManager:
    """Prevent oscillation with tier-downgrade delays and tier-specific cooldowns."""

    def __init__(self):
        self.current_tier: int = 1
        self.tier_duration: int = 0         # Consecutive intervals at current tier
        self.proposed_downgrade_count: int = 0  # Intervals proposing downgrade
        self.last_scale_up_time: float = 0.0
        self.last_scale_down_time: float = 0.0
        self.scale_up_cooldown_s: float = SCALE_UP_COOLDOWN_S

    def apply(self, proposed_tier: int, proposed_replicas: int,
              current_replicas: int) -> tuple[int, int]:
        """Return (tier, replicas) after downgrade persistence and scale-down cooldowns."""
        now = time.time()

        if proposed_tier < self.current_tier:
            self.proposed_downgrade_count += 1
            if self.proposed_downgrade_count < TIER_DOWNGRADE_DELAY:
                proposed_tier = self.current_tier
            else:
                # Allow downgrade after delay
                self.proposed_downgrade_count = 0
        elif proposed_tier > self.current_tier:
            self.proposed_downgrade_count = 0
        else:
            # Same tier
            self.proposed_downgrade_count = 0

        self.current_tier = proposed_tier

        final_replicas = proposed_replicas

        if proposed_replicas > current_replicas:
            # Scale-up: enforce cooldown
            if self.last_scale_up_time > 0 and (now - self.last_scale_up_time) < self.scale_up_cooldown_s:
                final_replicas = current_replicas
            else:
                self.last_scale_up_time = now
        elif proposed_replicas < current_replicas:
            # Scale-down: enforce tier-specific cooldown
            cooldown_s = _tier_cooldown(self.current_tier)
            if self.last_scale_down_time > 0 and (now - self.last_scale_down_time) < cooldown_s:
                final_replicas = current_replicas
            else:
                self.last_scale_down_time = now

        return self.current_tier, final_replicas
