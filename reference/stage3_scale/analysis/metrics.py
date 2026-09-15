"""Metrics: compute primary and secondary experimental metrics."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .loader import load_runs

logger = logging.getLogger(__name__)


SLO_TARGET_P95_MS = 200.0  # p95 < 200ms E2E latency (matches profiler + collector defaults)
DEFAULT_CAPACITY_RPS = 10.0  # RPS per replica at complexity=50000


def compute_slo_violation_rate(
    df: pd.DataFrame,
    slo_threshold_ms: float = SLO_TARGET_P95_MS,
) -> pd.DataFrame:
    """Ensure slo_violation_rate column exists, computing from p95 if needed."""
    df = df.copy()

    if "slo_violation_rate" in df.columns:
        mask = df["slo_violation_rate"].isna()
        if mask.any():
            logger.info("Computing SLO violation rate for %d runs with NaN values", mask.sum())
            if "slo_violation_intervals" in df.columns and "total_intervals" not in df.columns:
                df.loc[mask, "slo_violation_rate"] = (
                    df.loc[mask, "slo_violation_intervals"]
                    / (df.loc[mask, "actual_duration_s"] / 15)
                )
    else:
        logger.warning("No slo_violation_rate column — computing from p95")
        df["slo_violation_rate"] = (df.get("p95_ms", np.nan) > slo_threshold_ms).astype(float)

    return df


def compute_efficiency_score(df: pd.DataFrame) -> pd.Series:
    """Compute efficiency score: (1 - violation_rate) / (overhead_hours + 1)."""
    violation = df["slo_violation_rate"].clip(0, 1)
    overhead_hours = df["overhead_replica_seconds"].fillna(0) / 3600.0
    return (1 - violation) / (overhead_hours + 1)


def aggregate_by_method_workload(
    df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute mean, std, and CI for key metrics grouped by method × workload."""
    if len(df) == 0:
        return pd.DataFrame()

    group_cols = ["method", "workload"]

    metric_cols = [
        "slo_violation_rate",
        "mean_replicas",
        "max_replicas",
        "replica_churn",
        "overhead_replica_seconds",
        "p95_ms",
        "p50_ms",
    ]

    # Only include metrics that exist
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        return pd.DataFrame()

    grouped = df.groupby(group_cols)

    # Build result manually to avoid pandas agg complexity
    results = []
    for (method, workload), group in grouped:
        row = {"method": method, "workload": workload}
        for col in available:
            values = group[col].dropna()
            n = len(values)
            mean = values.mean()
            std = values.std(ddof=1) if n > 1 else 0.0
            sem = std / np.sqrt(n) if n > 1 else 0.0
            t_crit = stats.t.ppf(1 - alpha / 2, n - 1) if n > 1 else 0.0
            ci_half = sem * t_crit

            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_n"] = n
            row[f"{col}_ci_lower"] = mean - ci_half
            row[f"{col}_ci_upper"] = mean + ci_half
        results.append(row)

    return pd.DataFrame(results)


def _inference_latency_per_method(input_dir: Path) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        return {}
    for cell in input_dir.iterdir():
        if not cell.is_dir():
            continue
        method = cell.name.split("_", 1)[0]
        if not method.startswith("confscale-"):
            continue
        scale_log = cell / "controller_scale_log.json"
        if not scale_log.exists():
            continue
        try:
            rows = json.loads(scale_log.read_text())
        except Exception as e:
            logger.warning("Failed to parse %s: %s", scale_log, e)
            continue
        latencies = [
            float(r["decision_latency_s"]) * 1000.0
            for r in rows
            if isinstance(r, dict) and r.get("decision_latency_s") is not None
        ]
        if latencies:
            out.setdefault(method, []).extend(latencies)
    return {m: float(np.mean(v)) for m, v in out.items() if v}


def compute_uq_comparison(
    df: pd.DataFrame,
    input_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Extract UQ-specific metrics for Table 3 (UQ method comparison)."""
    uq_df = df[df["method"].str.startswith("confscale-")].copy()

    if len(uq_df) == 0:
        return pd.DataFrame()

    # Efficiency score is a run-level metric, aggregate from the runs frame
    uq_df["efficiency_score"] = compute_efficiency_score(uq_df)
    efficiency = uq_df.groupby("method")["efficiency_score"].mean()
    n_replicates = uq_df.groupby("method").size().rename("n_replicates")

    coverage = pd.Series(dtype=float, name="coverage")
    coverage_std = pd.Series(dtype=float, name="coverage_std")
    median_width = pd.Series(dtype=float, name="median_ci_width")
    mean_width = pd.Series(dtype=float, name="mean_ci_width")
    inference_ms = pd.Series(dtype=float, name="inference_ms")

    if input_dir is not None:
        try:
            from .calibration import analyze_all_cells
            per_cell = analyze_all_cells(Path(input_dir))
        except Exception as e:
            logger.warning("Calibration analysis failed for %s: %s", input_dir, e)
            per_cell = pd.DataFrame()
        if not per_cell.empty:
            agg = per_cell.groupby("method").agg(
                coverage=("empirical_coverage", "mean"),
                coverage_std=("empirical_coverage", "std"),
                median_ci_width=("median_width", "mean"),
                mean_ci_width=("mean_width", "mean"),
            )
            coverage = agg["coverage"]
            coverage_std = agg["coverage_std"]
            median_width = agg["median_ci_width"]
            mean_width = agg["mean_ci_width"]

        lat = _inference_latency_per_method(Path(input_dir))
        if lat:
            inference_ms = pd.Series(lat, name="inference_ms")

    # Fallback: legacy metrics.json columns (typically NaN today)
    if coverage.empty and "coverage" in uq_df.columns:
        coverage = uq_df.groupby("method")["coverage"].mean().rename("coverage")
        coverage_std = uq_df.groupby("method")["coverage"].std().rename("coverage_std")
    if mean_width.empty and "mean_ci_width" in uq_df.columns:
        mean_width = uq_df.groupby("method")["mean_ci_width"].mean().rename("mean_ci_width")
    if inference_ms.empty and "inference_ms" in uq_df.columns:
        inference_ms = uq_df.groupby("method")["inference_ms"].mean().rename("inference_ms")

    out = pd.concat(
        [coverage, coverage_std, median_width, mean_width,
         efficiency.rename("efficiency_score"),
         inference_ms, n_replicates],
        axis=1,
    )
    out.index.name = "method"
    return out


def compute_lambda_sweep(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Extract lambda sweep data if any exists (E3 experiments)."""
    if "lambda" in df.columns:
        sweep = df.copy()
    else:
        logger.info("No lambda column found — skipping lambda sweep")
        return None

    return sweep.groupby("lambda").agg(
        slo_violation_rate=("slo_violation_rate", "mean"),
        slo_violation_rate_std=("slo_violation_rate", "std"),
        overhead_replica_seconds=("overhead_replica_seconds", "mean"),
        mean_replicas=("mean_replicas", "mean"),
        n=("replicate", "count"),
    ).sort_index().reset_index()


def compute_summary(df: pd.DataFrame) -> dict:
    """Compute a comprehensive summary of all experiment results."""
    if len(df) == 0:
        return {"error": "No data"}

    df = compute_slo_violation_rate(df)

    return {
        "n_runs": len(df),
        "n_methods": df["method"].nunique(),
        "n_workloads": df["workload"].nunique(),
        "n_failed": len(df[df["status"].isin(["failed", "timeout"])]),
        "overall_mean_slo_rate": df["slo_violation_rate"].mean(),
        "overall_mean_replicas": df["mean_replicas"].mean(),
        "best_method": df.groupby("method")["slo_violation_rate"].mean().idxmin(),
        "best_method_rate": df.groupby("method")["slo_violation_rate"].mean().min(),
        "methods": df.groupby("method")["slo_violation_rate"].mean().to_dict(),
        "per_workload_best": (
            df.groupby(["workload", "method"])["slo_violation_rate"]
            .mean().groupby("workload").idxmin().to_dict()
        ),
    }


SUPPORTED_COMMANDS = ("load_runs", "aggregate_by_method_workload", "compute_uq_comparison", "compute_summary")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print(f"Usage: python metrics.py <input_dir> [command]")
        print(f"Commands: {', '.join(SUPPORTED_COMMANDS)}")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    command = sys.argv[2] if len(sys.argv) > 2 else "compute_summary"

    df = load_runs(input_dir)
    df = compute_slo_violation_rate(df)

    if command == "compute_summary":
        import json
        print(json.dumps(compute_summary(df), indent=2, default=str))
    elif command == "aggregate_by_method_workload":
        agg = aggregate_by_method_workload(df)
        print(agg.to_string())
    elif command == "compute_uq_comparison":
        uq = compute_uq_comparison(df, input_dir=input_dir)
        print(uq.to_string())
    else:
        print(f"Unknown command: {command}")
