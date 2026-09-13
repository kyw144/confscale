#!/usr/bin/env python3
"""
Smoke validation for the E1 Conformal PID recalibrator.

Standalone unit test for the ``ConformalPID`` class in
``uq/conformal_pid.py``. No cluster, Prometheus, or trained model
required — we feed synthetic residual streams and check the α-tracking
behavior, gain ablations, sign-convention edge cases, and clipping.

Run:
  python test_conformal_pid.py
or:
  pytest test_conformal_pid.py -v
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

from uq.aci import ACI  # noqa: E402
from uq.conformal_pid import (  # noqa: E402
    ConformalPID,
    EmptyResidualBufferError,
)


PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    results.append((label, condition, detail))
    glyph = PASS if condition else FAIL
    print(f"  {glyph} {label}" + (f"  — {detail}" if detail else ""))


def run_feedback(
    pid: ConformalPID,
    residuals: np.ndarray,
    warmup_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-fill buffer, then run residuals through the recalibration loop.

    Returns ``(alphas, miscovered_flags)`` for each post-warmup cycle. The
    cycle is: read ``quantile()`` → check ``|r| > Q`` → ``update(r, miss)``.
    """
    for r in residuals[:warmup_size]:
        pid.residuals.append(abs(float(r)))

    alphas = []
    miscovered = []
    for r in residuals[warmup_size:]:
        q = pid.quantile()
        miss = abs(float(r)) > q
        pid.update(float(r), miss)
        alphas.append(pid.alpha)
        miscovered.append(miss)
    return np.array(alphas), np.array(miscovered)


# ── Tests ───────────────────────────────────────────────────────────────


def test_defaults_and_init():
    print("\n[1] Construction defaults")
    pid = ConformalPID()
    check("target_alpha default 0.1", pid.target_alpha == 0.1)
    check("k_p default 0.1", pid.k_p == 0.1)
    check("k_i default 0.01", pid.k_i == 0.01)
    check("k_d default 0.05", pid.k_d == 0.05)
    check("alpha initialized to target", pid.alpha == 0.1)
    check("integral starts at 0", pid.integral == 0.0)
    check("buffer starts empty", len(pid.residuals) == 0)
    check("steps starts at 0", pid.steps == 0)

    pid2 = ConformalPID(target_alpha=0.2, alpha_init=0.35)
    check("alpha_init honored", pid2.alpha == 0.35)


def test_invalid_args():
    print("\n[2] Invalid constructor args rejected")
    for bad in [-0.1, 0.0, 1.0, 1.5]:
        try:
            ConformalPID(target_alpha=bad)
            check(f"target_alpha={bad} raises", False, "no exception")
        except ValueError:
            check(f"target_alpha={bad} raises ValueError", True)
    for bad in [0, -1]:
        try:
            ConformalPID(residual_buffer_size=bad)
            check(f"residual_buffer_size={bad} raises", False, "no exception")
        except ValueError:
            check(f"residual_buffer_size={bad} raises ValueError", True)
    # Inverted clip
    try:
        ConformalPID(alpha_clip=(0.5, 0.1))
        check("inverted alpha_clip raises", False, "no exception")
    except ValueError:
        check("inverted alpha_clip raises ValueError", True)


def test_empty_buffer_quantile_raises():
    print("\n[3] quantile() on empty buffer raises")
    pid = ConformalPID()
    try:
        pid.quantile()
        check("EmptyResidualBufferError raised", False, "no exception")
    except EmptyResidualBufferError:
        check("EmptyResidualBufferError raised", True)


def test_sign_convention_miss_widens():
    print("\n[4] Sign convention: a miss lowers α (widens next interval)")
    pid = ConformalPID(target_alpha=0.1, k_p=0.1, k_i=0.0, k_d=0.0,
                       alpha_init=0.1)
    alpha_before = pid.alpha
    pid.residuals.append(1.0)  # avoid empty buffer
    pid.update(residual=2.0, miscovered=True)
    check("alpha decreased after miss",
          pid.alpha < alpha_before,
          f"before={alpha_before:.4f} after={pid.alpha:.4f}")

    pid2 = ConformalPID(target_alpha=0.1, k_p=0.1, k_i=0.0, k_d=0.0,
                        alpha_init=0.1)
    alpha_before2 = pid2.alpha
    pid2.residuals.append(1.0)
    pid2.update(residual=0.5, miscovered=False)
    check("alpha increased after cover",
          pid2.alpha > alpha_before2,
          f"before={alpha_before2:.4f} after={pid2.alpha:.4f}")


def test_static_when_all_gains_zero():
    print("\n[5] All gains zero → α never moves")
    pid = ConformalPID(k_p=0.0, k_i=0.0, k_d=0.0, alpha_init=0.15)
    rng = np.random.default_rng(0)
    for r in rng.standard_normal(100):
        pid.residuals.append(abs(float(r)))
        pid.update(float(r), miscovered=bool(rng.random() < 0.3))
    check("alpha unchanged after 100 random updates",
          abs(pid.alpha - 0.15) < 1e-12,
          f"got {pid.alpha}")


def test_pid_reduces_to_aci_when_ki_kd_zero():
    print("\n[6] PID(k_i=k_d=0) trajectory matches ACI(η=k_p)")
    rng = np.random.default_rng(7)
    residuals = rng.standard_normal(200)
    # Generate miscovered flags from a fixed Bernoulli stream so both
    # objects see identical inputs.
    flags = (rng.random(200) < 0.15)

    pid = ConformalPID(target_alpha=0.1, k_p=0.2, k_i=0.0, k_d=0.0)
    aci = ACI(target_alpha=0.1, eta=0.2)
    for r, miss in zip(residuals, flags):
        pid.update(float(r), bool(miss))
        aci.update(float(r), bool(miss))
    check("final α exactly equal",
          pid.alpha == aci.alpha,
          f"pid={pid.alpha} aci={aci.alpha}")
    check("integral identical (both 0 since k_i=0 used)",
          pid.integral == aci.integral,
          f"pid={pid.integral} aci={aci.integral}")


def test_alpha_clip_upper():
    print("\n[7] α clipped to upper bound under sustained covers")
    pid = ConformalPID(k_p=0.5, k_i=0.1, k_d=0.0, alpha_clip=(0.05, 0.3),
                       alpha_init=0.1)
    for _ in range(500):
        pid.residuals.append(1.0)
        pid.update(0.5, miscovered=False)  # every step pushes α up
    check("α clipped at upper bound 0.3",
          pid.alpha == 0.3, f"got {pid.alpha}")


def test_alpha_clip_lower():
    print("\n[8] α clipped to lower bound under sustained misses")
    pid = ConformalPID(k_p=0.5, k_i=0.1, k_d=0.0, alpha_clip=(0.05, 0.3),
                       alpha_init=0.2)
    for _ in range(500):
        pid.residuals.append(1.0)
        pid.update(10.0, miscovered=True)  # every step pushes α down
    check("α clipped at lower bound 0.05",
          pid.alpha == 0.05, f"got {pid.alpha}")


def test_quantile_finite_sample_correction():
    print("\n[9] quantile() uses ceil((1-α)(n+1))-th order statistic")
    pid = ConformalPID(target_alpha=0.1, k_p=0.0, k_i=0.0, k_d=0.0,
                       alpha_init=0.1)
    # Insert 9 residuals: |0|, 1, 2, ..., 8
    for v in range(9):
        pid.residuals.append(float(v))
    # n=9, α=0.1: q_index = ceil(0.9*10) - 1 = ceil(9) - 1 = 8 (= 9th value = 8.0)
    q = pid.quantile()
    check("90% conformal quantile of 0..8 == 8.0",
          q == 8.0, f"got {q}")

    # 20 residuals: 0, 1, ..., 19. α=0.1: q_index = ceil(0.9*21)-1 = 19-1 = 18 → residuals[18] = 18.0
    pid2 = ConformalPID(target_alpha=0.1, k_p=0.0, k_i=0.0, k_d=0.0)
    for v in range(20):
        pid2.residuals.append(float(v))
    q2 = pid2.quantile()
    check("90% conformal quantile of 0..19 == 18.0",
          q2 == 18.0, f"got {q2}")


def test_stationary_convergence_tracks_target():
    print("\n[10] Stationary stream — α and empirical coverage track target")
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(450)
    pid = ConformalPID(target_alpha=0.1)
    alphas, miscovered = run_feedback(pid, residuals, warmup_size=100)
    # Time-averaged α over the last 200 cycles should hover near target.
    mean_alpha_late = float(alphas[-200:].mean())
    emp_miscoverage_late = float(miscovered[-200:].mean())
    check("mean α over last 200 cycles within ±0.03 of target",
          abs(mean_alpha_late - 0.1) < 0.03,
          f"mean α={mean_alpha_late:.4f}")
    # Empirical miscoverage = binomial(p≈target, n=200) — SE≈0.021. Allow ±0.06.
    check("empirical miscoverage over last 200 within ±0.06 of target",
          abs(emp_miscoverage_late - 0.1) < 0.06,
          f"emp miscov={emp_miscoverage_late:.4f}")


def test_recovery_under_volatility_break():
    print("\n[11] Volatility break — coverage rate recovers within ~50 cycles")
    rng = np.random.default_rng(1)
    # Phase 1: low variance for 200 cycles (buffer + initial run).
    phase1 = rng.standard_normal(200)
    # Phase 2: 3× variance jump for 150 cycles.
    phase2 = 3.0 * rng.standard_normal(150)
    residuals = np.concatenate([phase1, phase2])

    pid = ConformalPID(target_alpha=0.1)
    alphas, miscovered = run_feedback(pid, residuals, warmup_size=100)
    # After warmup (idx 0) the break occurs at cycle 100 (post-warmup index).
    # Pre-break miscoverage should hover near target; post-break+50 should
    # have recovered.
    pre_break = float(miscovered[50:100].mean())
    immediate_post = float(miscovered[100:120].mean())
    recovered = float(miscovered[150:].mean())
    check("pre-break miscoverage near target (±0.07)",
          abs(pre_break - 0.1) < 0.07, f"pre={pre_break:.4f}")
    check("immediate post-break miscoverage spikes above target",
          immediate_post > pre_break,
          f"pre={pre_break:.4f} post={immediate_post:.4f}")
    check("miscoverage recovers within 50 cycles (within ±0.10 of target)",
          abs(recovered - 0.1) < 0.10,
          f"recovered={recovered:.4f}")


def test_recovery_under_level_drift():
    print("\n[12] Level-drift — coverage rate recovers after mean shift")
    rng = np.random.default_rng(2)
    phase1 = rng.standard_normal(200)
    phase2 = rng.standard_normal(150) + 2.5  # mean ramp
    residuals = np.concatenate([phase1, phase2])

    pid = ConformalPID(target_alpha=0.1)
    alphas, miscovered = run_feedback(pid, residuals, warmup_size=100)
    immediate_post = float(miscovered[100:120].mean())
    recovered = float(miscovered[200:].mean())
    check("immediate post-shift miscoverage spikes",
          immediate_post > 0.15,
          f"post-shift={immediate_post:.4f}")
    check("miscoverage recovers within ±0.10 of target",
          abs(recovered - 0.1) < 0.10,
          f"recovered={recovered:.4f}")


def test_state_keys():
    print("\n[13] state() returns expected keys")
    pid = ConformalPID()
    expected = {
        "alpha", "target_alpha", "integral", "derivative",
        "last_error", "steps", "buffer_size", "k_p", "k_i", "k_d",
    }
    got = set(pid.state().keys())
    check("state() keys match", got == expected,
          f"missing {expected - got} extra {got - expected}")


def test_symmetric_input_no_drift():
    print("\n[14] Symmetric input stream — α stays near target")
    # Alternating cover/miss/cover/miss... at exact target rate (10 misses
    # per 100 cycles) should keep α near target. We do NOT use the feedback
    # loop because that would defeat the symmetry — we pass flags directly.
    pid = ConformalPID(target_alpha=0.1)
    rng = np.random.default_rng(3)
    # Build a stream of 90 covers + 10 misses, shuffled, repeated 5 times.
    pattern = np.array([False] * 90 + [True] * 10)
    rng.shuffle(pattern)
    stream = np.tile(pattern, 5)
    residuals = np.abs(rng.standard_normal(len(stream)))
    for r, miss in zip(residuals, stream):
        pid.residuals.append(float(r))
        pid.update(float(r), bool(miss))
    check("final α within ±0.05 of target on exact-rate stream",
          abs(pid.alpha - 0.1) < 0.05,
          f"final α={pid.alpha:.4f}")


# ── Main ────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.WARNING)
    print("E1 ConformalPID validation")
    print("=" * 60)

    test_defaults_and_init()
    test_invalid_args()
    test_empty_buffer_quantile_raises()
    test_sign_convention_miss_widens()
    test_static_when_all_gains_zero()
    test_pid_reduces_to_aci_when_ki_kd_zero()
    test_alpha_clip_upper()
    test_alpha_clip_lower()
    test_quantile_finite_sample_correction()
    test_stationary_convergence_tracks_target()
    test_recovery_under_volatility_break()
    test_recovery_under_level_drift()
    test_state_keys()
    test_symmetric_input_no_drift()

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
