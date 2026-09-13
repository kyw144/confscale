#!/usr/bin/env python3
"""
Smoke validation for the E1 coverage-conditional EscalationLadder.

Covers: trigger exactly at escalation_persistence, de-escalation exactly
at recovery_persistence, no level-skipping downward, oscillating coverage
must not accumulate spurious escalation, apply() factors per level, and
state-snapshot keys.

Run:
  python test_escalation_ladder.py
or:
  pytest test_escalation_ladder.py -v
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from baselines.escalation_ladder import EscalationLadder  # noqa: E402


PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    results.append((label, condition, detail))
    glyph = PASS if condition else FAIL
    print(f"  {glyph} {label}" + (f"  — {detail}" if detail else ""))


def step_n(ladder: EscalationLadder, coverage: float, n: int) -> list[int]:
    """Step the ladder n times at constant coverage, return level history."""
    return [ladder.step(coverage) for _ in range(n)]


# ── Tests ───────────────────────────────────────────────────────────────


def test_construction_defaults():
    print("\n[1] Construction defaults")
    L = EscalationLadder()
    check("target_coverage default 0.9", L.target_coverage == 0.9)
    check("escalation_band default 0.05", L.escalation_band == 0.05)
    check("escalation_persistence default 5", L.escalation_persistence == 5)
    check("recovery_persistence default 10", L.recovery_persistence == 10)
    check("widening_factor default 1.5", L.widening_factor == 1.5)
    check("initial level == 0", L.level == 0)
    check("escalation_threshold = 0.85",
          abs(L.escalation_threshold - 0.85) < 1e-9,
          f"got {L.escalation_threshold}")


def test_invalid_args():
    print("\n[2] Invalid constructor args rejected")
    for bad in [0.0, -0.1, 1.5]:
        try:
            EscalationLadder(target_coverage=bad)
            check(f"target_coverage={bad} raises", False, "no exception")
        except ValueError:
            check(f"target_coverage={bad} raises ValueError", True)
    for bad in [0, -1]:
        try:
            EscalationLadder(escalation_persistence=bad)
            check(f"escalation_persistence={bad} raises", False, "no exception")
        except ValueError:
            check(f"escalation_persistence={bad} raises ValueError", True)
    try:
        EscalationLadder(widening_factor=0.5)
        check("widening_factor<1 raises", False, "no exception")
    except ValueError:
        check("widening_factor<1 raises ValueError", True)


def test_triggers_exactly_at_escalation_persistence():
    print("\n[3] Level 0→1 fires exactly at escalation_persistence")
    L = EscalationLadder(target_coverage=0.9, escalation_band=0.05,
                          escalation_persistence=5)
    # Below threshold (0.85) for 4 cycles → still at level 0
    levels = step_n(L, 0.80, 4)
    check("4 below-band cycles → level still 0",
          levels == [0, 0, 0, 0], f"got {levels}")
    # 5th below-band cycle → level 1
    level5 = L.step(0.80)
    check("5th below-band cycle → level 1",
          level5 == 1, f"got {level5}")


def test_no_early_or_late_escalation():
    print("\n[4] No early escalation; no second escalation without persistence")
    L = EscalationLadder(escalation_persistence=5)
    # 4 below, then 1 in-band, then 4 below → still 0 (counter reset by in-band)
    for _ in range(4):
        L.step(0.80)
    L.step(0.87)  # in middle band, neither below nor at-target → resets
    levels = step_n(L, 0.80, 4)
    check("4 below + 1 mid + 4 below → still 0",
          L.level == 0 and levels == [0, 0, 0, 0],
          f"final={L.level} trail={levels}")


def test_oscillating_coverage_does_not_escalate():
    print("\n[5] Oscillating coverage doesn't accumulate spurious escalation")
    L = EscalationLadder(escalation_persistence=5)
    # Alternate below-band and above-target for 30 cycles
    seen_levels = []
    for i in range(30):
        cov = 0.80 if i % 2 == 0 else 0.95
        seen_levels.append(L.step(cov))
    check("never escalated despite many below-band cycles",
          all(lv == 0 for lv in seen_levels),
          f"max level seen={max(seen_levels)}")


def test_escalates_only_one_level_per_persistence_run():
    print("\n[6] 0→1→2 requires two full persistence runs")
    L = EscalationLadder(escalation_persistence=5)
    levels = step_n(L, 0.80, 5)
    check("0→1 after first run", L.level == 1, f"got {L.level}")
    # Second run of 5 below-band cycles to escalate 1→2
    step_n(L, 0.80, 4)
    check("still 1 after 4 more below-band cycles", L.level == 1)
    L.step(0.80)
    check("1→2 after 5th cycle of second run", L.level == 2)
    # Third run: try to escalate beyond 2 — should clamp
    step_n(L, 0.80, 20)
    check("MAX_LEVEL clamp at 2", L.level == 2, f"got {L.level}")


def test_deescalation_only_after_recovery_persistence():
    print("\n[7] De-escalation requires exactly recovery_persistence cycles")
    L = EscalationLadder(escalation_persistence=5, recovery_persistence=10)
    step_n(L, 0.80, 5)
    assert L.level == 1
    # 9 above-target cycles → still 1
    step_n(L, 0.95, 9)
    check("still 1 after 9 above-target cycles", L.level == 1, f"got {L.level}")
    # 10th cycle → 0
    L.step(0.95)
    check("1→0 after 10th cycle", L.level == 0, f"got {L.level}")


def test_no_level_skipping_downward():
    print("\n[8] De-escalation goes 2→1→0, not 2→0 directly")
    L = EscalationLadder(escalation_persistence=5, recovery_persistence=10)
    step_n(L, 0.80, 5)
    step_n(L, 0.80, 5)
    assert L.level == 2
    # 10 above-target cycles → 1, not 0
    step_n(L, 0.95, 10)
    check("2→1 after 10 recovery cycles", L.level == 1, f"got {L.level}")
    # Another 10 → 0
    step_n(L, 0.95, 10)
    check("1→0 after another 10 recovery cycles", L.level == 0, f"got {L.level}")


def test_recovery_counter_resets_on_below_band():
    print("\n[9] Recovery counter resets if coverage drops below band")
    L = EscalationLadder(escalation_persistence=5, recovery_persistence=10)
    step_n(L, 0.80, 5)
    assert L.level == 1
    # 9 above-target + 1 below-band → recovery counter resets, no de-escalation
    step_n(L, 0.95, 9)
    L.step(0.80)
    check("level still 1 after recovery interrupted",
          L.level == 1, f"got {L.level}")
    # Now we need another full 10 above-target cycles
    step_n(L, 0.95, 9)
    check("still 1 after 9 more", L.level == 1, f"got {L.level}")
    L.step(0.95)
    check("1→0 after 10th", L.level == 0, f"got {L.level}")


def test_apply_factors():
    print("\n[10] apply() multiplies by the correct level factor")
    L = EscalationLadder(widening_factor=1.5, conservative_factor=3.0)
    check("level 0: pass-through", L.apply(10.0) == 10.0)
    step_n(L, 0.80, 5)
    check("level 1: ×1.5",
          abs(L.apply(10.0) - 15.0) < 1e-9, f"got {L.apply(10.0)}")
    step_n(L, 0.80, 5)
    check("level 2: ×3.0",
          abs(L.apply(10.0) - 30.0) < 1e-9, f"got {L.apply(10.0)}")


def test_state_keys():
    print("\n[11] state() returns expected keys")
    L = EscalationLadder()
    expected = {
        "level", "escalation_counter", "recovery_counter",
        "escalation_threshold", "target_coverage",
        "escalation_persistence", "recovery_persistence",
        "widening_factor", "conservative_factor",
    }
    got = set(L.state().keys())
    check("state() keys match", got == expected,
          f"missing={expected - got} extra={got - expected}")


def test_brief_acceptance_pattern_h():
    print("\n[12] Brief acceptance: fires L1 within 5 of onset, de-escalates within 15 of recovery")
    L = EscalationLadder(escalation_persistence=5, recovery_persistence=10)
    # Onset: 5 below-band cycles
    for i in range(5):
        lvl = L.step(0.70)
        if lvl >= 1:
            check(f"Level 1 fired at cycle {i+1} (within 5)", i + 1 <= 5)
            break
    else:
        check("Level 1 fired within 5 cycles of onset", False,
              "never escalated")
    # Recovery: at least 10 above-target cycles brings level back to 0
    for i in range(15):
        lvl = L.step(0.95)
        if lvl == 0:
            check(f"De-escalated to Level 0 by cycle {i+1} (within 15)",
                  i + 1 <= 15)
            break
    else:
        check("De-escalated within 15 cycles of recovery", False,
              f"final level={L.level}")


# ── Main ────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.WARNING)
    print("E1 EscalationLadder validation")
    print("=" * 60)

    test_construction_defaults()
    test_invalid_args()
    test_triggers_exactly_at_escalation_persistence()
    test_no_early_or_late_escalation()
    test_oscillating_coverage_does_not_escalate()
    test_escalates_only_one_level_per_persistence_run()
    test_deescalation_only_after_recovery_persistence()
    test_no_level_skipping_downward()
    test_recovery_counter_resets_on_below_band()
    test_apply_factors()
    test_state_keys()
    test_brief_acceptance_pattern_h()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  {passed}/{total} checks passed")
    if passed != total:
        print("\n  FAILED:")
        for label, ok, detail in results:
            if not ok:
                print(f"    {FAIL} {label}  — {detail}")
        sys.exit(1)
    print(f"  {PASS} ALL GREEN")


if __name__ == "__main__":
    main()
