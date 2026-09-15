#!/usr/bin/env python3
"""SLO Model Fitting — fit the quadratic SLO model to profiling data."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SLO_TARGET_MS = 200


def slo_model(r_over_p: np.ndarray, L_base: float, a: float, b: float) -> np.ndarray:
    """Quadratic SLO model: L = L_base + a * (r/p) + b * (r/p)²"""
    return L_base + a * r_over_p + b * r_over_p**2


def fit_slo_model(rps_values: list[float], p95_values: list[float],
                  replicas: int = 1,
                  min_rps_for_fit: float = 5.0) -> dict:
    """Fit the quadratic SLO model to (rps, p95_latency) data."""
    valid = [
        (r, l) for r, l in zip(rps_values, p95_values)
        if r is not None and l is not None and r >= min_rps_for_fit
    ]
    if len(valid) < 4:
        logger.warning("Only %d valid data points (need ≥4 for quadratic fit)", len(valid))
        return {"error": "insufficient_data", "n_points": len(valid)}

    rps, p95 = zip(*valid)
    r_over_p = np.array(rps) / replicas

    try:
        from scipy.optimize import curve_fit
    except ImportError:
        logger.error("scipy not available — cannot fit model")
        return {"error": "scipy_unavailable"}

    # Initial guess: base latency from lowest RPS point, linear extrapolation
    L_base_init = min(p95)
    a_init = 0.1
    b_init = 0.001

    try:
        params, cov = curve_fit(
            slo_model, r_over_p, np.array(p95),
            p0=[L_base_init, a_init, b_init],
            maxfev=10000,
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),  # Constrain L_base ≥ 0, a ≥ 0, b ≥ 0
        )
        L_base, a, b = params

        predicted = slo_model(r_over_p, L_base, a, b)
        ss_res = np.sum((np.array(p95) - predicted) ** 2)
        ss_tot = np.sum((np.array(p95) - np.mean(p95)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except Exception as e:
        logger.warning("Quadratic fit failed: %s — trying linear fallback", e)
        # Linear fallback: b = 0
        try:
            def linear_model(x, L_base, a):
                return L_base + a * x
            params, cov = curve_fit(
                linear_model, r_over_p, np.array(p95),
                p0=[L_base_init, a_init], maxfev=10000
            )
            L_base, a = params
            b = 0.0
            predicted = linear_model(r_over_p, L_base, a)
            ss_res = np.sum((np.array(p95) - predicted) ** 2)
            ss_tot = np.sum((np.array(p95) - np.mean(p95)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            logger.info("Linear fallback: L_base=%.2f, a=%.4f, R²=%.3f", L_base, a, r_squared)
        except Exception as e2:
            logger.error("Linear fallback also failed: %s", e2)
            return {"error": "fit_failed"}

    # Handle negative b (physically impossible — saturation should be positive)
    if b < 0:
        logger.info("Negative quadratic term (b=%.6f) — clamping to 0 (linear model)", b)
        b = 0.0

    try:
        c = L_base - SLO_TARGET_MS
        if abs(b) < 1e-10:
            # Linear: a*r + c = 0 => r = -c/a
            r_single_replica_capacity = float(-c / a) if a > 0 else float('inf')
        else:
            discriminant = a**2 - 4 * b * c
            if discriminant < 0:
                r_single_replica_capacity = float('inf')  # Never saturates at this complexity
            else:
                roots = np.roots([b, a, c])
                r_single_replica_capacity = float(max(r for r in roots if r > 0))
    except Exception:
        r_single_replica_capacity = float('inf')

    result = {
        "L_base_ms": round(float(L_base), 2),
        "a": round(float(a), 6),
        "b": round(float(b), 8),
        "r_single_replica_capacity": round(r_single_replica_capacity, 1),
        "r_squared": round(float(r_squared), 4),
        "n_points": len(valid),
        "model_type": "linear" if abs(b) < 1e-10 else "quadratic",
    }

    logger.info(
        "Fit: L_base=%.2fms, a=%.4f, b=%.6f, R²=%.3f, capacity=%.1f RPS",
        L_base, a, b, r_squared, r_single_replica_capacity
    )
    return result


def compute_scaling_linearity(scaling_csv: Path) -> Optional[float]:
    """Compute scaling linearity from P2 data."""
    if not scaling_csv.exists():
        logger.warning("Scaling CSV not found: %s", scaling_csv)
        return None

    import csv
    rows = []
    with open(scaling_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if len(rows) < 2:
        return None

    efficiencies = []
    for r in rows:
        replicas = int(r["replicas"])
        total_req = float(r.get("total_requests", 0))
        # Each test is 120s
        actual_rps = total_req / 120.0 if total_req > 0 else float(r.get("target_rps", 0))
        target = float(r["target_rps"])
        # Efficiency: if target_rps / replicas ~= base_rps, efficiency ~1.0
        # Use the smallest replica count as baseline
        eff = actual_rps / target if target > 0 else 1.0
        efficiencies.append(eff)

    return round(float(np.mean(efficiencies)), 3) if efficiencies else None


def extract_step_response_timing(step_csv: Path) -> Optional[dict]:
    """Extract step-response timing."""
    if not step_csv.exists():
        logger.warning("Step response CSV not found: %s", step_csv)
        return None

    import csv
    events = []
    with open(step_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            events.append(r)

    cold_start_s = None
    warm_up_s = None
    for e in events:
        if e["event"] == "new_pod_running" and cold_start_s is None:
            cold_start_s = float(e["elapsed_s"])
        if e["event"] == "p95_under_slo" and warm_up_s is None:
            warm_up_s = float(e["elapsed_s"])

    if cold_start_s and warm_up_s:
        return {
            "cold_start_seconds": round(cold_start_s, 1),
            "warm_up_seconds": round(warm_up_s - cold_start_s, 1),
            "time_to_slo_seconds": round(warm_up_s, 1),
        }
    return None


def build_service_model(capacity_csv: Path,
                        scaling_csv: Optional[Path] = None,
                        step_csv: Optional[Path] = None,
                        bottleneck_result: Optional[dict] = None,
                        complexity: int = 100000) -> dict:
    """Build the complete service_model.json from all profiling outputs."""
    import csv

    rps_list = []
    p95_list = []
    with open(capacity_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("complexity", 0)) == complexity:
                try:
                    actual = float(row["actual_rps"]) if row.get("actual_rps") else 0
                    target = float(row.get("target_rps", 0))
                    l = float(row["p95_ms"]) if row.get("p95_ms") else None
                    # Filter out deeply saturated points: actual_rps / target_rps < 0.7
                    # These points are dominated by queueing, not the service model
                    if actual > 0 and target > 0 and l is not None:
                        throughput_ratio = actual / target
                        if throughput_ratio >= 0.7 and l < 2000:  # Don't fit through deep saturation
                            rps_list.append(actual)
                            p95_list.append(l)
                except (ValueError, TypeError):
                    continue

    if not rps_list:
        logger.error("No valid points found for complexity=%d in %s", complexity, capacity_csv)
        return {"error": "no_data"}

    fit = fit_slo_model(rps_list, p95_list)
    if "error" in fit:
        return fit

    compute_worker = {
        "L_base_ms": fit["L_base_ms"],
        "a": fit["a"],
        "b": fit["b"],
        "r_single_replica_capacity": fit["r_single_replica_capacity"],
        "r_squared": fit["r_squared"],
        "fitted_on": Path(capacity_csv).stat().st_mtime if capacity_csv.exists() else None,
        "complexity": complexity,
        "n_points": fit["n_points"],
        "model_type": fit["model_type"],
    }

    if scaling_csv and scaling_csv.exists():
        linearity = compute_scaling_linearity(scaling_csv)
        if linearity is not None:
            compute_worker["scaling_linearity"] = linearity

    if step_csv and step_csv.exists():
        timing = extract_step_response_timing(step_csv)
        if timing:
            compute_worker.update(timing)

    model = {
        "compute-worker": compute_worker,
        "frontend": {
            "max_rps_per_replica": 500.0,
            "verified_not_bottleneck": False,
        },
        "processor": {
            "max_rps_per_replica": 200.0,
            "verified_not_bottleneck": False,
        },
    }

    if bottleneck_result:
        if "frontend" in bottleneck_result:
            model["frontend"]["verified_not_bottleneck"] = bottleneck_result["frontend"].get(
                "verified_not_bottleneck", False
            )
            if "max_cpu_observed" in bottleneck_result["frontend"]:
                model["frontend"]["max_cpu_observed"] = bottleneck_result["frontend"]["max_cpu_observed"]
        if "processor" in bottleneck_result:
            model["processor"]["verified_not_bottleneck"] = bottleneck_result["processor"].get(
                "verified_not_bottleneck", False
            )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fit SLO model to profiling data"
    )
    parser.add_argument("--capacity-csv",
                        default="outputs/profiling/capacity_curve.csv",
                        help="Path to capacity_curve.csv")
    parser.add_argument("--scaling-csv",
                        default="outputs/profiling/scaling_linearity.csv",
                        help="Path to scaling_linearity.csv")
    parser.add_argument("--step-csv",
                        default="outputs/profiling/step_response.csv",
                        help="Path to step_response.csv")
    parser.add_argument("--output", default="outputs/profiling/service_model.json",
                        help="Output path for service_model.json")
    parser.add_argument("--complexity", type=int, default=100000,
                        help="Complexity level to fit model for")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    capacity_csv = Path(args.capacity_csv)
    if not capacity_csv.exists():
        logger.error("Capacity curve CSV not found: %s", args.capacity_csv)
        sys.exit(1)

    model = build_service_model(
        capacity_csv=capacity_csv,
        scaling_csv=Path(args.scaling_csv) if args.scaling_csv else None,
        step_csv=Path(args.step_csv) if args.step_csv else None,
        complexity=args.complexity,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "compute-worker" in model and model["compute-worker"].get("fitted_on"):
        from datetime import datetime
        model["compute-worker"]["fitted_on"] = datetime.fromtimestamp(
            model["compute-worker"]["fitted_on"]
        ).strftime("%Y-%m-%d")

    output_path.write_text(json.dumps(model, indent=2, default=str))
    logger.info("Wrote service_model.json to %s", output_path)

    cw = model.get("compute-worker", {})
    print(f"\nSLO Model Summary (complexity={cw.get('complexity', '?')})")
    print(f"  L_base = {cw.get('L_base_ms', '?')}ms")
    print(f"  a = {cw.get('a', '?')}")
    print(f"  b = {cw.get('b', '?')}")
    print(f"  R² = {cw.get('r_squared', '?')}")
    print(f"  Single-replica capacity = {cw.get('r_single_replica_capacity', '?')} RPS")
    print(f"  Scaling linearity = {cw.get('scaling_linearity', 'N/A')}")
    print(f"  Cold start = {cw.get('cold_start_seconds', 'N/A')}s")
    print(f"  Frontend bottleneck = {model.get('frontend', {}).get('verified_not_bottleneck', '?')}")
    print(f"  Processor bottleneck = {model.get('processor', {}).get('verified_not_bottleneck', '?')}")


if __name__ == "__main__":
    main()
