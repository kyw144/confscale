#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent  # src/stage3_scale
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from analysis.loader import _slo_metric_columns  # noqa: E402

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _FAILURES.append(name)


def isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def test_with_e2e_present() -> None:
    print("test_with_e2e_present")
    m = {
        "e2e": {"p95_ms": 1105.2, "p50_ms": 409.5, "slo_violation_rate": 1.0,
                "slo_violation_intervals": 1730, "total_intervals": 1730},
        "slo": {"p95_ms": 726.8, "p50_ms": 367.7, "violation_rate": 1.0},
    }
    c = _slo_metric_columns(m)
    check("metric_basis == 'e2e'", c["metric_basis"] == "e2e", c["metric_basis"])
    check("bare p95_ms is the e2e value (1105.2)", c["p95_ms"] == 1105.2, str(c["p95_ms"]))
    check("p95_ms_e2e == 1105.2", c["p95_ms_e2e"] == 1105.2)
    check("p95_ms_controller == proxy 726.8 (explicit)", c["p95_ms_controller"] == 726.8)
    check("bare slo_violation_rate is e2e (1.0)", c["slo_violation_rate"] == 1.0)
    check("slo_violation_rate_controller present (1.0)", c["slo_violation_rate_controller"] == 1.0)


def test_without_e2e_no_silent_proxy() -> None:
    print("test_without_e2e_no_silent_proxy")
    m = {  # the bug condition: e2e block stripped, only the controller proxy survives
        "slo": {"p95_ms": 737.0, "p50_ms": 369.2, "violation_rate": 0.34,
                "violation_intervals": 82},
    }
    c = _slo_metric_columns(m)
    check("metric_basis == 'e2e_MISSING'", c["metric_basis"] == "e2e_MISSING", c["metric_basis"])
    check("bare p95_ms is NaN (NOT silently the proxy 737.0)",
          isnan(c["p95_ms"]), str(c["p95_ms"]))
    check("p95_ms_e2e is NaN", isnan(c["p95_ms_e2e"]))
    check("bare slo_violation_rate is NaN (NOT silently the proxy 0.34)",
          isnan(c["slo_violation_rate"]), str(c["slo_violation_rate"]))
    check("proxy still available explicitly: p95_ms_controller == 737.0",
          c["p95_ms_controller"] == 737.0)
    check("proxy still available explicitly: slo_violation_rate_controller == 0.34",
          c["slo_violation_rate_controller"] == 0.34)


def main() -> int:
    test_with_e2e_present()
    test_without_e2e_no_silent_proxy()
    print()
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}): {_FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
