"""Escalate interval width after persistent undercoverage."""

from __future__ import annotations


class EscalationLadder:
    """Three-level coverage-conditional escalation state machine."""

    LEVEL_NOMINAL = 0
    LEVEL_WIDENING = 1
    LEVEL_CONSERVATIVE = 2
    MAX_LEVEL = LEVEL_CONSERVATIVE

    def __init__(
        self,
        target_coverage: float = 0.9,
        escalation_band: float = 0.05,
        escalation_persistence: int = 5,
        recovery_persistence: int = 10,
        widening_factor: float = 1.5,
        conservative_factor: float = 3.0,
    ) -> None:
        if not (0.0 < target_coverage <= 1.0):
            raise ValueError(
                f"target_coverage must be in (0, 1], got {target_coverage}"
            )
        if not (0.0 <= escalation_band < target_coverage):
            raise ValueError(
                f"escalation_band must be in [0, target_coverage), got {escalation_band}"
            )
        if escalation_persistence <= 0:
            raise ValueError(
                f"escalation_persistence must be positive, got {escalation_persistence}"
            )
        if recovery_persistence <= 0:
            raise ValueError(
                f"recovery_persistence must be positive, got {recovery_persistence}"
            )
        if widening_factor < 1.0:
            raise ValueError(
                f"widening_factor must be >= 1.0, got {widening_factor}"
            )
        if conservative_factor < widening_factor:
            raise ValueError(
                f"conservative_factor ({conservative_factor}) must be >= "
                f"widening_factor ({widening_factor})"
            )

        self.target_coverage = float(target_coverage)
        self.escalation_band = float(escalation_band)
        self.escalation_persistence = int(escalation_persistence)
        self.recovery_persistence = int(recovery_persistence)
        self.widening_factor = float(widening_factor)
        self.conservative_factor = float(conservative_factor)

        self.level: int = self.LEVEL_NOMINAL
        self.escalation_counter: int = 0
        self.recovery_counter: int = 0

    @property
    def escalation_threshold(self) -> float:
        """Trailing coverage strictly below this triggers escalation counting."""
        return self.target_coverage - self.escalation_band

    def step(self, trailing_coverage: float) -> int:
        """Advance one cycle; recover one level at a time and reset counters in the middle band."""
        tc = float(trailing_coverage)

        if tc < self.escalation_threshold:
            self.escalation_counter += 1
            self.recovery_counter = 0
        elif tc >= self.target_coverage:
            self.recovery_counter += 1
            self.escalation_counter = 0
        else:
            self.escalation_counter = 0
            self.recovery_counter = 0

        if (
            self.escalation_counter >= self.escalation_persistence
            and self.level < self.MAX_LEVEL
        ):
            self.level += 1
            self.escalation_counter = 0
            self.recovery_counter = 0
        elif (
            self.recovery_counter >= self.recovery_persistence
            and self.level > self.LEVEL_NOMINAL
        ):
            self.level -= 1
            self.escalation_counter = 0
            self.recovery_counter = 0

        return self.level

    def apply(self, base_half_width: float) -> float:
        """Transform the recalibrator's half-width per the current level."""
        h = float(base_half_width)
        if self.level == self.LEVEL_NOMINAL:
            return h
        if self.level == self.LEVEL_WIDENING:
            return h * self.widening_factor
        return h * self.conservative_factor

    def state(self) -> dict:
        return {
            "level": int(self.level),
            "escalation_counter": int(self.escalation_counter),
            "recovery_counter": int(self.recovery_counter),
            "escalation_threshold": float(self.escalation_threshold),
            "target_coverage": float(self.target_coverage),
            "escalation_persistence": int(self.escalation_persistence),
            "recovery_persistence": int(self.recovery_persistence),
            "widening_factor": float(self.widening_factor),
            "conservative_factor": float(self.conservative_factor),
        }
