#!/usr/bin/env python3
"""
Smoke validation for the C3 error-monitored baseline.

Tests the standalone ErrorMonitor class without needing a kind cluster,
Prometheus, or the GRU predictor at inference time. Covers the three
scenarios called out in the Paper 3 reframe brief:

  - Low-error: trailing MAE stays under threshold → not elevated.
  - High-error: trailing MAE exceeds threshold → elevated.
  - Volatility drift: zero-mean residuals with high variance still
    elevate the trigger, because MAE averages absolute residuals.

Run:
  python test_error_monitor.py
or:
  pytest test_error_monitor.py -v
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from baselines.controller import ErrorMonitor  # noqa: E402


PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    results.append((label, condition, detail))
    glyph = PASS if condition else FAIL
    print(f"  {glyph} {label}" + (f"  — {detail}" if detail else ""))


def feed(monitor: ErrorMonitor, residuals: list[float]) -> None:
    """Push a list of signed residuals through the monitor.

    We construct (forecast, observed) pairs where observed = forecast - r,
    so |forecast - observed| = |r|. Forecast is fixed at 50 — its value
    is irrelevant since only the absolute difference matters.
    """
    for r in residuals:
        monitor.record(forecast=50.0, observed=50.0 - r)


# ── Tests ───────────────────────────────────────────────────────────────


def test_empty_monitor():
    print("\n[1] Empty monitor")
    m = ErrorMonitor(window=20)
    check("samples()==0", m.samples() == 0)
    check("mae()==0.0", m.mae() == 0.0)
    check("is_elevated(any) returns False",
          m.is_elevated(0.0) is False and m.is_elevated(100.0) is False)


def test_record_returns_abs_residual():
    print("\n[2] record() returns absolute residual")
    m = ErrorMonitor(window=20)
    r1 = m.record(forecast=10.0, observed=4.0)
    r2 = m.record(forecast=10.0, observed=14.0)
    check("|10 - 4| == 6.0", r1 == 6.0, f"got {r1}")
    check("|10 - 14| == 4.0", r2 == 4.0, f"got {r2}")
    check("samples()==2", m.samples() == 2)


def test_low_error_not_elevated():
    print("\n[3] Low-error scenario (residuals ~2, threshold 3)")
    rng = np.random.default_rng(42)
    m = ErrorMonitor(window=20)
    # 20 residuals jittered around 2 with small noise
    feed(m, [2.0 + rng.normal(0, 0.2) for _ in range(20)])
    mae = m.mae()
    check("MAE ≈ 2 (within tolerance)", 1.5 < mae < 2.5, f"mae={mae:.3f}")
    check("not elevated at threshold=3", not m.is_elevated(3.0),
          f"mae={mae:.3f}")
    check("elevated at threshold=1.5", m.is_elevated(1.5),
          f"mae={mae:.3f}")


def test_high_error_elevated():
    print("\n[4] High-error scenario (residuals ~15, threshold 3)")
    rng = np.random.default_rng(7)
    m = ErrorMonitor(window=20)
    feed(m, [15.0 + rng.normal(0, 1.0) for _ in range(20)])
    mae = m.mae()
    check("MAE ≈ 15 (within tolerance)", 13.0 < mae < 17.0,
          f"mae={mae:.3f}")
    check("elevated at threshold=3", m.is_elevated(3.0),
          f"mae={mae:.3f}")
    check("still elevated at threshold=10", m.is_elevated(10.0))


def test_volatility_drift_elevated():
    print("\n[5] Volatility-drift scenario (zero-mean, high variance, threshold 5)")
    # Symmetric ±10 residuals: mean(residual) == 0 but mean(|residual|) == 10.
    # This is the crux test — the trigger fires on variance because the
    # monitor averages absolute values, not signed values.
    pattern = [+10.0, -10.0] * 10  # length 20
    m = ErrorMonitor(window=20)
    feed(m, pattern)
    raw_mean = sum(pattern) / len(pattern)
    mae = m.mae()
    check("signed mean of residuals is ~0",
          abs(raw_mean) < 1e-9, f"raw_mean={raw_mean}")
    check("MAE == 10.0 (mean of |±10|)",
          abs(mae - 10.0) < 1e-9, f"mae={mae:.3f}")
    check("elevated at threshold=5",
          m.is_elevated(5.0), f"mae={mae:.3f}")


def test_window_truncation():
    print("\n[6] Window truncates to maxlen")
    m = ErrorMonitor(window=5)
    feed(m, [100.0] * 5)
    check("MAE == 100 after 5 high residuals",
          abs(m.mae() - 100.0) < 1e-9)
    # Now push 5 zero residuals — they should evict the 100s.
    feed(m, [0.0] * 5)
    check("samples()==5 (maxlen enforced)", m.samples() == 5,
          f"got {m.samples()}")
    check("MAE drops to 0.0 once window flushed",
          m.mae() == 0.0, f"mae={m.mae():.3f}")


def test_invalid_window():
    print("\n[7] Invalid window size rejected")
    for bad in [0, -1, -100]:
        try:
            ErrorMonitor(window=bad)
            check(f"ErrorMonitor(window={bad}) raises ValueError", False,
                  "no exception raised")
        except ValueError:
            check(f"ErrorMonitor(window={bad}) raises ValueError", True)


# ── Main ────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.WARNING)
    print("C3 ErrorMonitor validation")
    print("=" * 60)

    test_empty_monitor()
    test_record_returns_abs_residual()
    test_low_error_not_elevated()
    test_high_error_elevated()
    test_volatility_drift_elevated()
    test_window_truncation()
    test_invalid_window()

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
