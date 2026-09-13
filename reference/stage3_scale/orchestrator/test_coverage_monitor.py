#!/usr/bin/env python3
"""
Smoke validation for the C2-Analyze online coverage monitor.

Standalone unit test for the ``CoverageMonitor`` class in
``orchestrator/controller.py``. No cluster, Prometheus, or UQ model
required — we feed synthetic (predict, validate) sequences and check
the trailing coverage / alert-state / lifetime-counter math.

Run:
  python test_coverage_monitor.py
or:
  pytest test_coverage_monitor.py -v
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from orchestrator.coverage_monitor import CoverageMonitor  # noqa: E402


PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    results.append((label, condition, detail))
    glyph = PASS if condition else FAIL
    print(f"  {glyph} {label}" + (f"  — {detail}" if detail else ""))


def feed(monitor: CoverageMonitor, samples: list[tuple[float, float, float]]) -> None:
    """Run a list of (ci_lower, ci_upper, observed) tuples through the monitor.

    Each sample records then immediately validates with the given
    observed RPS, mimicking the controller's iteration-by-iteration
    flow (compressed: record + validate same call for test ergonomics).
    """
    for lo, hi, obs in samples:
        monitor.record_prediction(lo, hi)
        monitor.validate_pending(obs)


# ── Tests ───────────────────────────────────────────────────────────────


def test_empty_monitor():
    print("\n[1] Empty monitor")
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    check("trailing_coverage == 1.0 on init",
          m.trailing_coverage == 1.0, f"got {m.trailing_coverage}")
    check("alert_state == 0 on init",
          m.alert_state == 0, f"got {m.alert_state}")
    check("coverage_shortfall == 0.0 on init",
          m.coverage_shortfall() == 0.0,
          f"got {m.coverage_shortfall()}")
    check("needs_recalibration() == False on init",
          m.needs_recalibration() is False)
    state = m.get_state()
    expected_keys = {'trailing_coverage', 'coverage_shortfall', 'alert_state',
                     'target_coverage', 'window_size', 'pending', 'validated',
                     'covered'}
    check("get_state() returns expected keys",
          set(state.keys()) == expected_keys,
          f"got {set(state.keys())}")


def test_record_then_validate_covered():
    print("\n[2] Record + validate, observed inside interval")
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    m.record_prediction(10.0, 20.0)
    check("pending grows to 1 after record_prediction",
          len(m.pending) == 1)
    result = m.validate_pending(15.0)
    check("validate_pending(15.0) returns True", result is True,
          f"got {result}")
    check("pending shrinks back to 0", len(m.pending) == 0)
    check("validated has 1 entry == True",
          len(m.validated) == 1 and m.validated[-1] is True)
    check("trailing_coverage == 1.0 after one cover",
          m.trailing_coverage == 1.0,
          f"got {m.trailing_coverage}")
    check("alert_state == 0 (nominal)", m.alert_state == 0)


def test_record_then_validate_missed():
    print("\n[3] Record + validate, observed outside interval")
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    m.record_prediction(10.0, 20.0)
    result = m.validate_pending(25.0)
    check("validate_pending(25.0) returns False",
          result is False, f"got {result}")
    check("trailing_coverage == 0.0 after one miss",
          m.trailing_coverage == 0.0,
          f"got {m.trailing_coverage}")
    check("alert_state == 2 (critical) after one miss",
          m.alert_state == 2, f"got {m.alert_state}")
    check("coverage_shortfall == 0.9 (= target)",
          abs(m.coverage_shortfall() - 0.9) < 1e-9,
          f"got {m.coverage_shortfall()}")
    check("needs_recalibration() == True at critical",
          m.needs_recalibration() is True)


def test_validate_without_pending_returns_none():
    print("\n[4] validate_pending with empty pending returns None")
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    result = m.validate_pending(50.0)
    check("returns None", result is None, f"got {result}")
    check("validated stays empty", len(m.validated) == 0)
    check("trailing stays at 1.0",
          m.trailing_coverage == 1.0, f"got {m.trailing_coverage}")


def test_interval_endpoints_are_inclusive():
    print("\n[5] Interval endpoints are inclusive ([ci_lower, ci_upper])")
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    m.record_prediction(10.0, 20.0)
    check("observed == ci_lower counts as covered",
          m.validate_pending(10.0) is True)
    m.record_prediction(10.0, 20.0)
    check("observed == ci_upper counts as covered",
          m.validate_pending(20.0) is True)
    m.record_prediction(10.0, 20.0)
    check("observed just below ci_lower is a miss",
          m.validate_pending(9.999) is False)


def test_alert_state_thresholds():
    print("\n[6] Alert-state bands (target=0.9)")
    # Build a window with a known number of covers/misses, then check
    # the band. target=0.9, critical threshold = 0.85 * 0.9 = 0.765.

    # Window of 10: 9 covers + 1 miss → trailing=0.9 → nominal
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    feed(m, [(10.0, 20.0, 15.0)] * 9 + [(10.0, 20.0, 50.0)])
    check("trailing=0.9 → alert_state=0 (nominal, exactly at target)",
          abs(m.trailing_coverage - 0.9) < 1e-9 and m.alert_state == 0,
          f"trailing={m.trailing_coverage}, alert={m.alert_state}")

    # Window of 10: 8 covers + 2 misses → trailing=0.8 → warning
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    feed(m, [(10.0, 20.0, 15.0)] * 8 + [(10.0, 20.0, 50.0)] * 2)
    check("trailing=0.8 → alert_state=1 (warning)",
          abs(m.trailing_coverage - 0.8) < 1e-9 and m.alert_state == 1,
          f"trailing={m.trailing_coverage}, alert={m.alert_state}")

    # Window of 10: 7 covers + 3 misses → trailing=0.7 → critical
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    feed(m, [(10.0, 20.0, 15.0)] * 7 + [(10.0, 20.0, 50.0)] * 3)
    check("trailing=0.7 → alert_state=2 (critical, below 0.85·T=0.765)",
          abs(m.trailing_coverage - 0.7) < 1e-9 and m.alert_state == 2,
          f"trailing={m.trailing_coverage}, alert={m.alert_state}")

    # Boundary at exactly 0.85 * T = 0.765: must be warning (closed left).
    # Use window=200 so we can hit 0.765 exactly: 153 covers / 200.
    m = CoverageMonitor(window_size=200, target_coverage=0.9)
    feed(m, [(10.0, 20.0, 15.0)] * 153 + [(10.0, 20.0, 50.0)] * 47)
    check("trailing=0.765 (= 0.85·T) → alert_state=1 (warning, closed left)",
          abs(m.trailing_coverage - 0.765) < 1e-9 and m.alert_state == 1,
          f"trailing={m.trailing_coverage}, alert={m.alert_state}")


def test_coverage_shortfall():
    print("\n[7] coverage_shortfall = max(0, target - trailing)")
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    # 5 covers + 5 misses → trailing=0.5 → shortfall=0.4
    feed(m, [(10.0, 20.0, 15.0)] * 5 + [(10.0, 20.0, 50.0)] * 5)
    check("trailing=0.5 → shortfall=0.4",
          abs(m.coverage_shortfall() - 0.4) < 1e-9,
          f"got {m.coverage_shortfall()}")

    # Trailing above target → shortfall clamps to 0
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    feed(m, [(10.0, 20.0, 15.0)] * 10)
    check("trailing=1.0 → shortfall=0.0 (clamped)",
          m.coverage_shortfall() == 0.0,
          f"got {m.coverage_shortfall()}")


def test_sliding_window_evicts_oldest():
    print("\n[8] Validated window evicts oldest FIFO")
    m = CoverageMonitor(window_size=3, target_coverage=0.9)

    # Push 3 misses → trailing=0.0
    feed(m, [(10.0, 20.0, 50.0)] * 3)
    check("3 misses → trailing=0.0", m.trailing_coverage == 0.0,
          f"got {m.trailing_coverage}")
    check("validated cap at window_size=3", len(m.validated) == 3)

    # Push 3 covers → window flushes misses → trailing=1.0
    feed(m, [(10.0, 20.0, 15.0)] * 3)
    check("3 covers evict 3 misses → trailing=1.0",
          m.trailing_coverage == 1.0,
          f"got {m.trailing_coverage}")
    check("validated still capped at 3", len(m.validated) == 3)
    check("alert_state back to 0 (nominal)", m.alert_state == 0)


def test_lifetime_counters_grow_unbounded():
    print("\n[9] Lifetime counters count past window")
    m = CoverageMonitor(window_size=5, target_coverage=0.9)
    # Push 20 covers + 10 misses (window only sees the last 5)
    feed(m, [(10.0, 20.0, 15.0)] * 20 + [(10.0, 20.0, 50.0)] * 10)
    check("lifetime_validated == 30", m.lifetime_validated == 30,
          f"got {m.lifetime_validated}")
    check("lifetime_covered == 20", m.lifetime_covered == 20,
          f"got {m.lifetime_covered}")
    # Window sees only the last 5 (all misses) → trailing=0.0
    check("trailing=0.0 (last 5 are misses)",
          m.trailing_coverage == 0.0, f"got {m.trailing_coverage}")


def test_needs_recalibration_only_critical():
    print("\n[10] needs_recalibration() True only when alert_state == 2")
    # Nominal
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    feed(m, [(10.0, 20.0, 15.0)] * 10)
    check("nominal: needs_recalibration() == False",
          m.needs_recalibration() is False)

    # Warning band (trailing=0.8 < target=0.9, but >= 0.85·T)
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    feed(m, [(10.0, 20.0, 15.0)] * 8 + [(10.0, 20.0, 50.0)] * 2)
    check("warning: needs_recalibration() == False (warning < critical)",
          m.needs_recalibration() is False and m.alert_state == 1,
          f"alert={m.alert_state}")

    # Critical band (trailing=0.5 < 0.85·T=0.765)
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    feed(m, [(10.0, 20.0, 15.0)] * 5 + [(10.0, 20.0, 50.0)] * 5)
    check("critical: needs_recalibration() == True",
          m.needs_recalibration() is True and m.alert_state == 2,
          f"alert={m.alert_state}")


def test_invalid_args():
    print("\n[11] Invalid constructor args rejected")
    for bad in [0, -1, -100]:
        try:
            CoverageMonitor(window_size=bad, target_coverage=0.9)
            check(f"CoverageMonitor(window_size={bad}) raises", False,
                  "no exception")
        except ValueError:
            check(f"CoverageMonitor(window_size={bad}) raises ValueError",
                  True)
    for bad in [0.0, -0.5, 1.5]:
        try:
            CoverageMonitor(window_size=10, target_coverage=bad)
            check(f"CoverageMonitor(target_coverage={bad}) raises",
                  False, "no exception")
        except ValueError:
            check(f"CoverageMonitor(target_coverage={bad}) raises ValueError",
                  True)


def test_recorded_at_attached():
    print("\n[12] record_prediction stamps recorded_at")
    import time
    m = CoverageMonitor(window_size=10, target_coverage=0.9)
    before = time.time()
    m.record_prediction(10.0, 20.0)
    after = time.time()
    lo, hi, ts = m.pending[0]
    check("pending tuple is (lo, hi, recorded_at)",
          lo == 10.0 and hi == 20.0 and isinstance(ts, float),
          f"got ({lo}, {hi}, {ts})")
    check("recorded_at is between before/after wall-clock",
          before <= ts <= after,
          f"before={before} ts={ts} after={after}")


# ── Main ────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.WARNING)
    print("C2-Analyze CoverageMonitor validation")
    print("=" * 60)

    test_empty_monitor()
    test_record_then_validate_covered()
    test_record_then_validate_missed()
    test_validate_without_pending_returns_none()
    test_interval_endpoints_are_inclusive()
    test_alert_state_thresholds()
    test_coverage_shortfall()
    test_sliding_window_evicts_oldest()
    test_lifetime_counters_grow_unbounded()
    test_needs_recalibration_only_critical()
    test_invalid_args()
    test_recorded_at_attached()

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
