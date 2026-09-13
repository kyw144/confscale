#!/usr/bin/env python3
"""
Smoke validation for the baseline-controller CoverageMonitor wiring.

Drives ``build_operator_metrics_summary(...)`` directly with synthetic
``CoverageMonitor`` state and a fixture ``scale_log``. No cluster,
Prometheus, or kube context required. Covers the three contracts in
the brief:

  1. Monitor disabled → no summary block (function returns None).
  2. Monitor enabled with empty scale_log → no summary (defensive).
  3. Monitor enabled with a known (interval, realised) sequence →
     ``coverage_rate`` matches the expected value, and all fields the
     post-reframe loader reads are present with the right types.

Run:
  python test_baseline_coverage_monitor.py
or:
  pytest test_baseline_coverage_monitor.py -v
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from baselines.controller import build_operator_metrics_summary  # noqa: E402
from orchestrator.coverage_monitor import CoverageMonitor  # noqa: E402


PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    results.append((label, condition, detail))
    glyph = PASS if condition else FAIL
    print(f"  {glyph} {label}" + (f"  — {detail}" if detail else ""))


def _make_scale_log(n: int, target_replicas_seq: list[int] | None = None) -> list[dict]:
    """Return n scale-log entries with target_replicas defaulting to 1."""
    if target_replicas_seq is None:
        target_replicas_seq = [1] * n
    assert len(target_replicas_seq) == n
    return [{'target_replicas': r, 'actual_replicas': r} for r in target_replicas_seq]


def _feed(monitor: CoverageMonitor, samples: list[tuple[float, float, float]]) -> None:
    """Apply (lo, hi, observed) tuples through record + validate."""
    for lo, hi, obs in samples:
        monitor.record_prediction(lo, hi)
        monitor.validate_pending(obs)


# ── Tests ───────────────────────────────────────────────────────────────


def test_monitor_disabled_returns_none():
    print("\n[1] Monitor disabled → no summary emitted")
    scale_log = _make_scale_log(10)
    result = build_operator_metrics_summary(
        coverage_monitor=None,
        scale_log=scale_log,
        mode='hpa-error-monitored',
        final_replicas=1,
    )
    check("returns None when coverage_monitor is None",
          result is None, f"got {result!r}")


def test_empty_scale_log_returns_none():
    print("\n[2] Empty scale_log → no summary emitted")
    monitor = CoverageMonitor(window_size=10, target_coverage=0.9)
    _feed(monitor, [(10.0, 20.0, 15.0)] * 3)
    result = build_operator_metrics_summary(
        coverage_monitor=monitor,
        scale_log=[],
        mode='hpa-qr-monitored',
        final_replicas=1,
    )
    check("returns None when scale_log is empty",
          result is None, f"got {result!r}")


def test_coverage_rate_matches_fixture():
    print("\n[3] coverage_rate matches expected value on fixture sequence")
    # 7 covers + 3 misses → lifetime_covered=7, lifetime_validated=10,
    # coverage_rate=0.7. Window=30, so all 10 land in-window and
    # trailing_coverage == 0.7 too.
    monitor = CoverageMonitor(window_size=30, target_coverage=0.9)
    samples = [(10.0, 20.0, 15.0)] * 7 + [(10.0, 20.0, 50.0)] * 3
    _feed(monitor, samples)
    scale_log = _make_scale_log(10, target_replicas_seq=[1, 1, 2, 2, 3, 3, 3, 2, 2, 1])
    summary = build_operator_metrics_summary(
        coverage_monitor=monitor,
        scale_log=scale_log,
        mode='hpa-qr-monitored',
        final_replicas=1,
    )
    check("summary is a dict", isinstance(summary, dict),
          f"got {type(summary).__name__}")
    check("mode field == hpa-qr-monitored",
          summary['mode'] == 'hpa-qr-monitored', f"got {summary['mode']}")
    check("total_intervals == len(scale_log)",
          summary['total_intervals'] == 10,
          f"got {summary['total_intervals']}")
    # target_replicas seq has 4 transitions: 1→2, 2→3, 3→2, 2→1.
    check("total_scale_operations counts transitions (=4)",
          summary['total_scale_operations'] == 4,
          f"got {summary['total_scale_operations']}")
    check("final_replicas reflects passed argument",
          summary['final_replicas'] == 1,
          f"got {summary['final_replicas']}")

    cov = summary['coverage_monitor']
    check("coverage_monitor.enabled == True",
          cov['enabled'] is True, f"got {cov['enabled']}")
    check("coverage_monitor.total_validated == 10",
          cov['total_validated'] == 10, f"got {cov['total_validated']}")
    check("coverage_monitor.total_covered == 7",
          cov['total_covered'] == 7, f"got {cov['total_covered']}")
    check("coverage_monitor.coverage_rate == 0.7",
          abs(cov['coverage_rate'] - 0.7) < 1e-9,
          f"got {cov['coverage_rate']}")
    check("coverage_monitor.coverage_rate is a float",
          isinstance(cov['coverage_rate'], float),
          f"got {type(cov['coverage_rate']).__name__}")
    check("coverage_monitor.coverage_rate is in [0, 1]",
          0.0 <= cov['coverage_rate'] <= 1.0,
          f"got {cov['coverage_rate']}")
    check("coverage_monitor.final_trailing_coverage == 0.7",
          abs(cov['final_trailing_coverage'] - 0.7) < 1e-9,
          f"got {cov['final_trailing_coverage']}")
    check("coverage_monitor.target_coverage == 0.9",
          cov['target_coverage'] == 0.9, f"got {cov['target_coverage']}")
    check("coverage_monitor.window_size == 30",
          cov['window_size'] == 30, f"got {cov['window_size']}")
    # 0.7 < 0.765 = 0.85*T → critical band
    check("coverage_monitor.final_alert_state == 2 (critical at trailing=0.7)",
          cov['final_alert_state'] == 2,
          f"got {cov['final_alert_state']}")


def test_zero_validations_returns_none_coverage_rate():
    print("\n[4] coverage_rate == None when monitor has zero validations")
    # Monitor created but never fed — lifetime_validated == 0.
    monitor = CoverageMonitor(window_size=10, target_coverage=0.9)
    scale_log = _make_scale_log(5)
    summary = build_operator_metrics_summary(
        coverage_monitor=monitor,
        scale_log=scale_log,
        mode='hpa-qr-monitored',
        final_replicas=1,
    )
    cov = summary['coverage_monitor']
    check("total_validated == 0", cov['total_validated'] == 0,
          f"got {cov['total_validated']}")
    check("coverage_rate is None (not 0.0) when nothing validated",
          cov['coverage_rate'] is None, f"got {cov['coverage_rate']!r}")


def test_schema_keys_match_orchestrator_contract():
    print("\n[5] coverage_monitor block has all keys the post-reframe loader reads")
    monitor = CoverageMonitor(window_size=30, target_coverage=0.9)
    _feed(monitor, [(10.0, 20.0, 15.0)] * 10)
    summary = build_operator_metrics_summary(
        coverage_monitor=monitor,
        scale_log=_make_scale_log(10),
        mode='hpa-qr-monitored',
        final_replicas=1,
    )
    cov = summary['coverage_monitor']
    expected_keys = {
        'enabled', 'final_trailing_coverage', 'final_alert_state',
        'total_validated', 'total_covered', 'coverage_rate',
        'target_coverage', 'window_size',
    }
    check("coverage_monitor key set matches orchestrator schema",
          set(cov.keys()) == expected_keys,
          f"diff: {set(cov.keys()) ^ expected_keys}")


# ── Main ────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.WARNING)
    print("Baseline-controller CoverageMonitor wiring")
    print("=" * 60)

    test_monitor_disabled_returns_none()
    test_empty_scale_log_returns_none()
    test_coverage_rate_matches_fixture()
    test_zero_validations_returns_none_coverage_rate()
    test_schema_keys_match_orchestrator_contract()

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
