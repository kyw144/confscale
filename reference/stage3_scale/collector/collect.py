#!/usr/bin/env python3
"""
Metrics Collector — extract structured timeseries from Prometheus after experiments.

Contract (for orchestrator):
    collect_metrics(prometheus_url, run_id, start_time, end_time,
                    method, workload, replicate, output_dir) -> dict

Handles:
    - NaN/inf values → sanitized to None in JSON
    - Prometheus connection errors → retry 3x with 5s backoff
    - Empty query results → warn, not crash
    - Partial data → flag in metadata
"""

# Local artifact reference entrypoint; cluster behavior is unverified.
if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")


import csv
import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .queries import SCALAR_QUERIES, SLO_QUERIES, TIMESERIES_QUERIES, UQ_QUERIES, compute_e2e_slo_metrics
from .snapshot import capture_snapshot

logger = logging.getLogger(__name__)


# ── Prometheus API Client ──────────────────────────────────────────────

class PrometheusClient:
    """Minimal Prometheus HTTP API client with retry logic."""

    def __init__(self, base_url: str, max_retries: int = 3, backoff_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_s = backoff_s

    def _request(self, endpoint: str, params: dict) -> dict:
        """Execute a Prometheus API request with retry on connection errors."""
        url = f"{self.base_url}/api/v1/{endpoint}?"
        url += urllib.parse.urlencode(params)
        last_error = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read())
                if data.get("status") != "success":
                    raise RuntimeError(f"Prometheus error: {data.get('error', 'unknown')}")
                return data
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last_error = e
                logger.warning("Prometheus request failed (attempt %d/%d): %s",
                               attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_s * (attempt + 1))
        raise ConnectionError(f"Prometheus unreachable after {self.max_retries} attempts: {last_error}")

    def query(self, promql: str) -> list[dict]:
        """Execute an instant query. Returns list of result dicts."""
        data = self._request("query", {"query": promql})
        return data["data"]["result"]

    def query_range(self, promql: str, start: float, end: float, step: str) -> list[dict]:
        """Execute a range query. Returns list of result dicts with 'values' arrays."""
        data = self._request("query_range", {
            "query": promql,
            "start": start,
            "end": end,
            "step": step,
        })
        return data["data"]["result"]


# ── Value Sanitization ─────────────────────────────────────────────────

def sanitize_value(value: Any) -> Optional[float]:
    """Convert a Prometheus value to a float, returning None for NaN/Inf."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


# ── Query Execution ────────────────────────────────────────────────────

def _execute_scalar_queries(client: PrometheusClient, queries: list[dict],
                            window_s: int) -> dict[str, Any]:
    """Execute all scalar instant queries and return a flat dict of results."""
    results = {}
    window = f"{window_s}s"
    for q in queries:
        if q.get("optional") and q.get("fallback") is None:
            # Optional UQ metrics — skip quietly if they don't exist
            continue
        try:
            promql = q["promql"].format(window=window, window_s=window_s)
            data = client.query(promql)
            if data and data[0].get("value"):
                raw = data[0]["value"][1]
                val = sanitize_value(raw)
            else:
                val = q["fallback"]
            # Apply transform if defined
            if val is not None and "transform" in q:
                val = eval(q["transform"])(val)
            results[q["output_key"]] = val
        except Exception as e:
            logger.warning("Scalar query '%s' failed: %s", q["name"], e)
            results[q["output_key"]] = q.get("fallback")
    return results


def _execute_timeseries_queries(client: PrometheusClient, queries: list[dict],
                                start: float, end: float) -> dict[str, list[tuple[float, float]]]:
    """Execute all timeseries range queries. Returns {column_name: [(timestamp, value), ...]}."""
    results = {}
    for q in queries:
        try:
            data = client.query_range(q["promql"], start, end, q["step"])
            if not data:
                logger.warning("Timeseries query '%s' returned no data", q["name"])
                results[q["csv_column"]] = []
                continue
            values = data[0].get("values", [])
            col = q["csv_column"]
            has_transform = "transform" in q and q.get("transform") is not None
            parsed = []
            for ts_str, val_str in values:
                ts = float(ts_str)
                v = sanitize_value(val_str)
                if v is not None and has_transform:
                    v = eval(q["transform"])(v)
                parsed.append((ts, v))
            results[col] = parsed
        except Exception as e:
            logger.warning("Timeseries query '%s' failed: %s", q["name"], e)
            results[q["csv_column"]] = []
    return results


# ── Output Writers ─────────────────────────────────────────────────────

def _write_metrics_json(scalars: dict, output_dir: Path, run_id: str,
                        method: str, workload: str, replicate: int,
                        duration_s: int, window_s: int) -> Path:
    """Write the summary metrics.json file."""
    # Nest the flat scalar keys into the structured schema
    structured: dict[str, Any] = {
        "run_id": run_id,
        "method": method,
        "workload": workload,
        "replicate": replicate,
        "duration_s": duration_s,
        "window_s": window_s,
    }
    # Build nested dicts from dotted keys: "slo.p95_ms" → {"slo": {"p95_ms": ...}}
    for key, val in scalars.items():
        parts = key.split(".")
        d = structured
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = val

    path = output_dir / "metrics.json"
    path.write_text(json.dumps(structured, indent=2, default=str))
    logger.info("Wrote metrics.json (%d bytes) to %s", path.stat().st_size, path)
    return path


def _write_timeseries_csv(ts_data: dict[str, list[tuple[float, float]]],
                          output_dir: Path, duration_s: int) -> Path:
    """Write the timeseries.csv file, aligning all columns by timestamp."""
    # Build a merged dict: {timestamp: {col: value, ...}}
    merged: dict[float, dict[str, Optional[float]]] = {}
    columns = list(ts_data.keys())
    for col in columns:
        for ts, val in ts_data[col]:
            if ts not in merged:
                merged[ts] = {c: None for c in columns}
            merged[ts][col] = val

    path = output_dir / "timeseries.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + columns)
        for ts in sorted(merged.keys()):
            row = [ts] + [merged[ts][col] for col in columns]
            writer.writerow(row)

    logger.info("Wrote timeseries.csv (%d rows, %d cols) to %s",
                len(merged), len(columns) + 1, path)
    return path


# ── Main Entry Point ───────────────────────────────────────────────────

def collect_metrics(
    prometheus_url: str,
    run_id: str,
    start_time: datetime,
    end_time: datetime,
    method: str,
    workload: str,
    replicate: int,
    output_dir: Path,
    do_snapshot: bool = True,
    snapshot_pod_name: str = "prometheus-prometheus-kube-prometheus-prometheus-0",
    trace_csv_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Extract metrics from Prometheus and write structured output.

    Args:
        prometheus_url: Prometheus HTTP API base URL (e.g., http://localhost:9090)
        run_id: Unique run identifier (e.g., hpa-reactive_diurnal_rep1_20260509_120000)
        start_time: Experiment start time (UTC)
        end_time: Experiment end time (UTC)
        method: Method name (e.g., hpa-reactive, confscale-be)
        workload: Workload pattern (e.g., diurnal, spike)
        replicate: Replicate number (1-indexed)
        output_dir: Directory to write metrics.json and timeseries.csv
        do_snapshot: If True, trigger and copy a Prometheus TSDB snapshot
        snapshot_pod_name: Prometheus pod name for kubectl cp

    Returns:
        dict with status, metadata, and summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = PrometheusClient(prometheus_url)

    # Convert datetimes to epoch seconds for Prometheus queries
    start_epoch = start_time.timestamp()
    end_epoch = end_time.timestamp()
    duration_s = int(end_epoch - start_epoch)
    window_s = duration_s  # Prometheus range window = experiment duration

    status = "ok"
    warnings: list[str] = []

    # ── 1. Execute scalar queries ──
    logger.info("Executing scalar queries...")
    scalar_results = _execute_scalar_queries(client, SCALAR_QUERIES, window_s)

    # ── 2. Execute SLO queries ──
    logger.info("Executing SLO queries...")
    slo_results = _execute_scalar_queries(client, SLO_QUERIES, window_s)
    scalar_results.update(slo_results)

    # ── 2b. Compute E2E SLO metrics from workload trace CSV ──
    if trace_csv_path and trace_csv_path.exists():
        logger.info("Computing E2E SLO metrics from trace: %s", trace_csv_path)
        e2e_metrics = compute_e2e_slo_metrics(trace_csv_path)
        scalar_results["e2e.p50_ms"] = e2e_metrics.get("e2e_p50_ms")
        scalar_results["e2e.p95_ms"] = e2e_metrics.get("e2e_p95_ms")
        scalar_results["e2e.p99_ms"] = e2e_metrics.get("e2e_p99_ms")
        scalar_results["e2e.slo_violation_rate"] = e2e_metrics.get("e2e_slo_violation_rate")
        scalar_results["e2e.slo_violation_intervals"] = e2e_metrics.get("e2e_slo_violation_intervals")
        scalar_results["e2e.total_intervals"] = e2e_metrics.get("e2e_total_intervals")
    elif trace_csv_path:
        logger.warning("Trace CSV not found at %s — skipping E2E SLO metrics", trace_csv_path)

    # ── 3. Execute UQ queries (optional) ──
    logger.info("Checking for UQ metrics...")
    try:
        uq_results = _execute_scalar_queries(client, UQ_QUERIES, window_s)
        # Only include UQ results if at least one metric was found
        if any(v is not None for v in uq_results.values()):
            scalar_results.update(uq_results)
    except Exception as e:
        logger.info("UQ metrics not available (expected for non-UQ runs): %s", e)

    # ── 4. Execute timeseries queries ──
    logger.info("Executing timeseries queries...")
    ts_results = _execute_timeseries_queries(
        client, TIMESERIES_QUERIES, start_epoch, end_epoch
    )

    # Compute derived metrics
    avg_replicas = scalar_results.pop("resources._avg_replicas_for_overhead", 1.0)
    if avg_replicas is not None:
        scalar_results["resources.overhead_replica_seconds"] = max(0, (avg_replicas - 1) * window_s)
    else:
        scalar_results["resources.overhead_replica_seconds"] = 0.0
    n_scalars = sum(1 for v in scalar_results.values() if v is not None)
    total_scalars = len(scalar_results)
    if n_scalars == 0:
        status = "degraded"
        warnings.append("All scalar queries returned no data")
    elif n_scalars < total_scalars:
        warnings.append(f"Partial scalar data: {n_scalars}/{total_scalars} metrics resolved")

    n_ts_points = sum(len(v) for v in ts_results.values())
    if n_ts_points == 0:
        if status == "ok":
            status = "degraded"
        warnings.append("Timeseries queries returned no datapoints")

    # ── 6. Write output ──
    _write_metrics_json(scalar_results, output_dir, run_id,
                        method, workload, replicate, duration_s, window_s)
    _write_timeseries_csv(ts_results, output_dir, duration_s)

    # ── 6. Optional TSDB snapshot ──
    snapshot_info = None
    if do_snapshot:
        logger.info("Capturing TSDB snapshot...")
        snapshot_info = capture_snapshot(
            prometheus_url, output_dir, pod_name=snapshot_pod_name
        )
        if not snapshot_info["success"]:
            warnings.append(f"TSDB snapshot failed: {snapshot_info.get('error')}")

    # ── 7. Check data quality ──
    summary = {
        "status": status,
        "warnings": warnings,
        "method": method,
        "workload": workload,
        "replicate": replicate,
        "duration_s": duration_s,
        "slo_violation_rate": scalar_results.get("slo.violation_rate"),
        "p95_ms": scalar_results.get("slo.p95_ms"),
        "e2e_slo_violation_rate": scalar_results.get("e2e.slo_violation_rate"),
        "e2e_p95_ms": scalar_results.get("e2e.p95_ms"),
        "mean_replicas": scalar_results.get("resources.mean_replicas"),
        "max_replicas": scalar_results.get("resources.max_replicas"),
        "total_requests": scalar_results.get("requests.total"),
        "error_rate": scalar_results.get("requests.error_rate"),
        "scalar_metrics_resolved": n_scalars,
        "scalar_metrics_total": total_scalars,
        "timeseries_datapoints": n_ts_points,
        "snapshot": snapshot_info,
    }
    return summary


# ── CLI (for standalone testing) ────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Extract metrics from Prometheus")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--start-time", required=True, help="ISO 8601 UTC timestamp")
    parser.add_argument("--end-time", required=True, help="ISO 8601 UTC timestamp")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start_time)
    end = datetime.fromisoformat(args.end_time)

    summary = collect_metrics(
        prometheus_url=args.prometheus_url,
        run_id=args.run_id,
        start_time=start,
        end_time=end,
        method=args.method,
        workload=args.workload,
        replicate=args.replicate,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, default=str))
