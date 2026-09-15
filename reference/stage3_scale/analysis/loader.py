"""Loader: read experiment run directories into structured DataFrames."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class RunMeta:
    """Metadata extracted from run directory name and contents."""
    run_id: str
    method: str
    workload: str
    replicate: int
    timestamp: str
    directory: Path
    status: str = "unknown"
    duration_s: int = 0
    actual_duration_s: int = 0
    error_message: str = ""

    metrics: dict = field(default_factory=dict)

    config: dict = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return self.directory.exists()

    @property
    def is_valid(self) -> bool:
        return self.status not in ("failed", "timeout", "unknown")


def _parse_run_dirname(dirname: str) -> Optional[dict]:
    parts = dirname.split("_")
    # Method names may contain underscores; locate the replicate marker first.
    try:
        rep_idx = None
        for i, p in enumerate(parts):
            if p.startswith("rep"):
                rep_idx = i
                break

        if rep_idx is None or rep_idx < 2:
            return None  # Can't parse

        # rep is at rep_idx, workload at rep_idx-1, timestamp after rep_idx
        method = "_".join(parts[:rep_idx - 1])
        workload = parts[rep_idx - 1]
        replicate = int(parts[rep_idx][3:])
        timestamp = "_".join(parts[rep_idx + 1:])

        return {
            "method": method,
            "workload": workload.upper() if len(workload) == 1 else workload,
            "replicate": replicate,
            "timestamp": timestamp,
        }
    except (ValueError, IndexError):
        return None


def discover_runs(input_dir: Path) -> list[RunMeta]:
    """Walk input_dir and discover all run directories."""
    input_dir = Path(input_dir)
    runs: list[RunMeta] = []

    for entry in sorted(input_dir.iterdir()):
        if not entry.is_dir():
            continue

        parsed = _parse_run_dirname(entry.name)
        if parsed is None:
            continue

        run_id = entry.name
        meta = RunMeta(
            run_id=run_id,
            method=parsed["method"],
            workload=parsed["workload"],
            replicate=parsed["replicate"],
            timestamp=parsed["timestamp"],
            directory=entry,
        )

        config_path = entry / "run_config.yaml"
        if config_path.exists():
            try:
                meta.config = yaml.safe_load(config_path.read_text()) or {}
                meta.status = meta.config.get("status", "unknown")
                meta.duration_s = meta.config.get("duration_s", 0)
                meta.actual_duration_s = meta.config.get("actual_duration_s", 0)
            except Exception as e:
                logger.warning("Failed to read config for %s: %s", run_id, e)

        metrics_path = entry / "metrics.json"
        if metrics_path.exists():
            try:
                meta.metrics = json.loads(metrics_path.read_text())
            except Exception as e:
                logger.warning("Failed to read metrics for %s: %s", run_id, e)

        runs.append(meta)

    logger.info("Discovered %d runs in %s", len(runs), input_dir)
    return runs


def _slo_metric_columns(m: dict) -> dict:
    e2e = m.get("e2e", {}) or {}
    slo = m.get("slo", {}) or {}
    has_e2e = e2e.get("p95_ms") is not None
    cols = {
        # explicit e2e basis (NaN if the block is absent — never proxy)
        "p50_ms_e2e": e2e.get("p50_ms", np.nan),
        "p95_ms_e2e": e2e.get("p95_ms", np.nan),
        "p99_ms_e2e": e2e.get("p99_ms", np.nan),
        "slo_violation_rate_e2e": e2e.get("slo_violation_rate", np.nan),
        "slo_violation_intervals_e2e": e2e.get("slo_violation_intervals", np.nan),
        "p50_ms_controller": slo.get("p50_ms", np.nan),
        "p95_ms_controller": slo.get("p95_ms", np.nan),
        "p99_ms_controller": slo.get("p99_ms", np.nan),
        "slo_violation_rate_controller": slo.get("violation_rate", np.nan),
        "slo_violation_intervals_controller": slo.get("violation_intervals", np.nan),
        "metric_basis": "e2e" if has_e2e else "e2e_MISSING",
    }
    # bare columns kept for backward-compat, but e2e-ONLY (no silent proxy fallback)
    cols["p50_ms"] = cols["p50_ms_e2e"]
    cols["p95_ms"] = cols["p95_ms_e2e"]
    cols["p99_ms"] = cols["p99_ms_e2e"]
    cols["slo_violation_rate"] = cols["slo_violation_rate_e2e"]
    cols["slo_violation_intervals"] = cols["slo_violation_intervals_e2e"]
    return cols


def load_runs(input_dir: Path) -> pd.DataFrame:
    """Load all runs into a flat DataFrame with one row per run."""
    runs = discover_runs(input_dir)
    rows = []

    for r in runs:
        m = r.metrics

        resources = m.get("resources", {})
        requests = m.get("requests", {})

        row = {
            "run_id": r.run_id,
            "method": r.method,
            "workload": r.workload,
            "replicate": r.replicate,
            "status": r.status,
            "duration_s": r.duration_s,
            "actual_duration_s": r.actual_duration_s,
            # SLO latency / violation columns — explicit per-basis, e2e never
            # silently backfilled with the controller proxy (E-V6b).
            **_slo_metric_columns(m),
            "mean_replicas": resources.get("mean_replicas", np.nan),
            "max_replicas": resources.get("max_replicas", np.nan),
            "replica_churn": resources.get("replica_churn", np.nan),
            "overhead_replica_seconds": resources.get("overhead_replica_seconds", np.nan),
            "total_requests": requests.get("total", np.nan),
            "error_rate": requests.get("error_rate", np.nan),
            "coverage": m.get("coverage", np.nan),
            "mean_ci_width": m.get("mean_ci_width", np.nan),
            "inference_ms": m.get("inference_ms", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        logger.warning("No runs found in %s", input_dir)
        return df

    # E-V6b: surface (never silently absorb) cells whose e2e block is missing —
    # their bare p95_ms / slo_violation_rate are NaN, not the controller proxy.
    if "metric_basis" in df.columns:
        n_missing = int((df["metric_basis"] == "e2e_MISSING").sum())
        if n_missing:
            logger.warning(
                "%d/%d runs lack an e2e.* block; p95_ms/slo_violation_rate are NaN "
                "for them (use *_controller for the proxy, or recover e2e). Affected: %s",
                n_missing, len(df),
                sorted(df.loc[df["metric_basis"] == "e2e_MISSING", "run_id"].tolist())[:10],
            )

    n_total = len(df)
    df_valid = df[df["status"].notna() & ~df["status"].isin(["failed", "timeout"])].copy()
    n_failed = n_total - len(df_valid)
    if n_failed > 0:
        logger.warning("Excluded %d failed/timeout runs from analysis", n_failed)

    logger.info("Loaded %d runs (%d valid, %d failed) from %s",
                 n_total, len(df_valid), n_failed, input_dir)
    return df_valid


def load_timeseries(input_dir: Path) -> pd.DataFrame:
    """Load all timeseries CSVs into a single DataFrame keyed by run_id."""
    runs = discover_runs(input_dir)
    frames = []

    for r in runs:
        ts_path = r.directory / "timeseries.csv"
        if not ts_path.exists():
            continue

        try:
            ts = pd.read_csv(ts_path)
            ts["run_id"] = r.run_id
            ts["method"] = r.method
            ts["workload"] = r.workload
            ts["replicate"] = r.replicate
            frames.append(ts)
        except Exception as e:
            logger.warning("Failed to read timeseries for %s: %s", r.run_id, e)

    if not frames:
        logger.warning("No timeseries data found")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d timeseries rows from %d runs", len(df), len(frames))
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Validate the loaded DataFrame for common issues."""
    warnings = []

    if len(df) == 0:
        return {"error": "No data loaded", "warnings": warnings}

    failed_runs = df[df["status"].isin(["failed", "timeout"])]
    if len(failed_runs) > 0:
        pct = 100 * len(failed_runs) / len(df)
        warnings.append(f"{len(failed_runs)} failed runs ({pct:.1f}%)")
        if pct > 5:
            warnings.append("⚠️ >5% of runs failed — experiment integrity may be compromised")

    for col in ["slo_violation_rate", "p95_ms", "mean_replicas"]:
        missing = df[col].isna().sum()
        if missing > 0:
            warnings.append(f"{missing} runs missing {col}")

    method_wl = df.groupby(["method", "workload"]).size()
    min_reps = method_wl.min() if len(method_wl) > 0 else 0
    if min_reps < 3:
        low_rep = method_wl[method_wl < 3]
        warnings.append(
            f"Low replicate coverage (min={min_reps}): "
            + ", ".join(f"{m}/{w}" for (m, w) in low_rep.index)
        )

    methods = df["method"].unique()
    workloads = df["workload"].unique()
    expected = len(methods) * len(workloads)
    if expected > 0:
        present = len(df.groupby(["method", "workload"]))
        if present < expected:
            warnings.append(
                f"Missing {expected - present}/{expected} method×workload combinations"
            )

    return {
        "n_runs": len(df),
        "n_methods": len(methods),
        "n_workloads": len(workloads),
        "n_combinations_present": present if expected > 0 else 0,
        "n_combinations_expected": expected,
        "warnings": warnings,
    }
