"""
PromQL query definitions for the metrics collector.

Each query is a dict with:
  - name: short identifier
  - promql: the PromQL query string (with {window} and {step} placeholders)
  - kind: 'scalar', 'vector', or 'timeseries'
  - description: human-readable
  - output_key: key name in the output structure
  - unit: measurement unit
  - fallback: value to use when query returns no data

Placeholders:
  {window}  — replaced with the experiment duration string (e.g., "3600s")
  {step}    — replaced with the step interval string (e.g., "15s")
"""

import logging

from pathlib import Path

logger = logging.getLogger(__name__)

# ── Scalar Queries (single values → summary) ──────────────────────────

SCALAR_QUERIES = [
    {
        "name": "total_requests",
        "promql": 'sum(increase(frontend_latency_seconds_count[{window}]))',
        "kind": "scalar",
        "description": "Total number of frontend requests during the experiment",
        "output_key": "requests.total",
        "unit": "count",
        "fallback": 0,
    },
    {
        "name": "error_count",
        "promql": 'sum(increase(flask_http_request_total{{status="504"}}[{window}]))',
        "kind": "scalar",
        "description": "Total 504 errors (frontend timeout on processor/compute)",
        "output_key": "requests.errors",
        "unit": "count",
        "fallback": 0,
    },
    {
        "name": "error_rate",
        "promql": (
            'sum(increase(flask_http_request_total{{status="504"}}[{window}]))'
            ' / '
            'sum(increase(frontend_latency_seconds_count[{window}]))'
        ),
        "kind": "scalar",
        "description": "Error rate as fraction of total requests",
        "output_key": "requests.error_rate",
        "unit": "fraction",
        "fallback": 0.0,
    },
    {
        "name": "mean_replicas",
        "promql": 'avg_over_time(kube_deployment_spec_replicas{{deployment="compute-worker"}}[{window}])',
        "kind": "scalar",
        "description": "Mean replica count over the experiment window",
        "output_key": "resources.mean_replicas",
        "unit": "float",
        "fallback": 0.0,
    },
    {
        "name": "max_replicas",
        "promql": 'max_over_time(kube_deployment_spec_replicas{{deployment="compute-worker"}}[{window}])',
        "kind": "scalar",
        "description": "Maximum replica count reached during the experiment",
        "output_key": "resources.max_replicas",
        "unit": "int",
        "fallback": 0,
    },
    {
        "name": "replica_churn",
        "promql": 'sum(changes(kube_deployment_spec_replicas{{deployment="compute-worker"}}[{window}]))',
        "kind": "scalar",
        "description": "Total number of replica count changes (scaling events)",
        "output_key": "resources.replica_churn",
        "unit": "count",
        "fallback": 0,
    },
    {
        "name": "overhead_replica_seconds",
        "promql": (
            'avg_over_time(kube_deployment_spec_replicas{{deployment="compute-worker"}}[{window}])'
        ),
        "kind": "scalar",
        "description": "Average replica count (overhead = (avg - 1) * window_s computed in Python)",
        "output_key": "resources._avg_replicas_for_overhead",
        "unit": "float",
        "fallback": 1.0,
        "note": "Actual overhead_replica_seconds = (val - 1) * window_s, computed in collect.py",
    },
    {
        "name": "mean_cpu_utilization",
        "promql": (
            'avg(avg_over_time('
            '  rate(container_cpu_usage_seconds_total{{container="compute"}}[1m])'
            '  [{window}:30s]'  # subquery resolution required by PromQL parser
            '))'
        ),
        "kind": "scalar",
        "description": "Mean CPU utilization averaged across all compute-worker pods",
        "output_key": "resources.mean_cpu_usage",
        "unit": "cores",
        "fallback": 0.0,
    },
    {
        "name": "mean_cpu_utilization_pct",
        "promql": (
            'avg(avg_over_time('
            '  rate(container_cpu_usage_seconds_total{{container="compute"}}[1m])'
            '  [{window}:30s]'  # subquery resolution required by PromQL parser
            '))'
            ' / '
            'avg(kube_pod_container_resource_limits{{resource="cpu",container="compute"}})'
        ),
        "kind": "scalar",
        "description": "Mean CPU utilization as fraction of CPU limit",
        "output_key": "resources.mean_cpu_utilization",
        "unit": "fraction",
        "fallback": 0.0,
    },
]

# ── SLO Queries (latency percentiles) ─────────────────────────────────

SLO_QUERIES = [
    {
        "name": "p50_latency_overall",
        "promql": (
            'histogram_quantile(0.50, '
            '  sum(rate(frontend_latency_seconds_bucket[{window}])) by (le)'
            ')'
        ),
        "kind": "scalar",
        "description": "Overall p50 frontend latency across all pods",
        "output_key": "slo.p50_ms",
        "unit": "ms",
        "fallback": None,
        "transform": "lambda v: v * 1000 if v is not None else None",
    },
    {
        "name": "p95_latency_overall",
        "promql": (
            'histogram_quantile(0.95, '
            '  sum(rate(frontend_latency_seconds_bucket[{window}])) by (le)'
            ')'
        ),
        "kind": "scalar",
        "description": "Overall p95 frontend latency across all pods",
        "output_key": "slo.p95_ms",
        "unit": "ms",
        "fallback": None,
        "transform": "lambda v: v * 1000 if v is not None else None",
    },
    {
        "name": "p99_latency_overall",
        "promql": (
            'histogram_quantile(0.99, '
            '  sum(rate(frontend_latency_seconds_bucket[{window}])) by (le)'
            ')'
        ),
        "kind": "scalar",
        "description": "Overall p99 frontend latency across all pods",
        "output_key": "slo.p99_ms",
        "unit": "ms",
        "fallback": None,
        "transform": "lambda v: v * 1000 if v is not None else None",
    },
    {
        "name": "slo_violation_rate",
        "promql": (
            'avg_over_time('
            '  (histogram_quantile(0.95, '
            '    sum(rate(frontend_latency_seconds_bucket[30s])) by (le)'
            '  ) > bool 0.2)[{window}:15s]'
            ')'
        ),
        "kind": "scalar",
        "description": "Fraction of 15s intervals where p95 latency exceeded 200ms SLO, averaged across all pods",
        "output_key": "slo.violation_rate",
        "unit": "fraction",
        "fallback": 0.0,
        "note": "Uses avg_over_time on subquery: returns 0-1 fraction of violating intervals",
    },
    {
        "name": "slo_violation_intervals",
        "promql": (
            'sum(sum_over_time('
            '  (histogram_quantile(0.95, '
            '    rate(frontend_latency_seconds_bucket[30s])'
            '  ) > bool 0.2)[{window}:15s]'
            '))'
        ),
        "kind": "scalar",
        "description": "Total count of 15s intervals (across all pods) where p95 exceeded 200ms SLO",
        "output_key": "slo.violation_intervals",
        "unit": "count",
        "fallback": 0,
        "note": "Uses sum(sum_over_time(...)) to aggregate across pods; total intervals = pods × (window/15s)",
    },
]


# ── Timeseries Queries (vector over time → timeseries.csv) ────────────

TIMESERIES_QUERIES = [
    {
        "name": "p95_latency_ts",
        "promql": (
            'histogram_quantile(0.95, '
            '  sum(rate(frontend_latency_seconds_bucket[30s])) by (le)'
            ')'
        ),
        "step": "15s",
        "description": "p95 frontend latency timeseries (aggregated across pods)",
        "csv_column": "p95_ms",
        "transform": "lambda v: v * 1000",
    },
    {
        "name": "p50_latency_ts",
        "promql": (
            'histogram_quantile(0.50, '
            '  sum(rate(frontend_latency_seconds_bucket[30s])) by (le)'
            ')'
        ),
        "step": "15s",
        "description": "p50 frontend latency timeseries (aggregated across pods)",
        "csv_column": "p50_ms",
        "transform": "lambda v: v * 1000",
    },
    {
        "name": "request_rate_ts",
        "promql": 'sum(rate(frontend_latency_seconds_count[30s]))',
        "step": "15s",
        "description": "Total request rate across all frontend pods",
        "csv_column": "rps",
        "transform": None,
    },
    {
        "name": "error_rate_ts",
        "promql": 'sum(rate(flask_http_request_total{status="504"}[30s]))',
        "step": "15s",
        "description": "Error rate (504s from frontend timeouts)",
        "csv_column": "error_rps",
        "transform": None,
    },
    {
        "name": "replicas_ts",
        "promql": 'kube_deployment_spec_replicas{deployment="compute-worker"}',
        "step": "15s",
        "description": "Compute-worker replica count",
        "csv_column": "replicas",
        "transform": None,
    },
    {
        "name": "cpu_usage_ts",
        "promql": 'sum(rate(container_cpu_usage_seconds_total{container="compute"}[30s]))',
        "step": "15s",
        "description": "Aggregate CPU usage across all compute-worker pods",
        "csv_column": "cpu_cores",
        "transform": None,
    },
    {
        "name": "slo_violation_ts",
        "promql": (
            'histogram_quantile(0.95, '
            '  sum(rate(frontend_latency_seconds_bucket[30s])) by (le)'
            ') > bool 0.2'
        ),
        "step": "15s",
        "description": "SLO violation indicator (1 if p95 > 200ms, else 0)",
        "csv_column": "violating",
        "transform": "lambda v: int(v)",
    },
]

# ── UQ-Specific Queries (for when prediction operator is running) ─────

UQ_QUERIES = [
    {
        "name": "predicted_rps",
        "promql": 'confidence_scaler_predicted_rps',
        "kind": "scalar",
        "description": "Predicted request rate from confidence scaler operator",
        "output_key": "predictions.rps",
        "unit": "rps",
        "fallback": None,
        "optional": True,
    },
    {
        "name": "ci_lower",
        "promql": 'confidence_scaler_ci_lower',
        "kind": "scalar",
        "description": "Lower bound of prediction interval",
        "output_key": "predictions.ci_lower",
        "unit": "rps",
        "fallback": None,
        "optional": True,
    },
    {
        "name": "ci_upper",
        "promql": 'confidence_scaler_ci_upper',
        "kind": "scalar",
        "description": "Upper bound of prediction interval",
        "output_key": "predictions.ci_upper",
        "unit": "rps",
        "fallback": None,
        "optional": True,
    },
    {
        "name": "confidence_tier",
        "promql": 'confidence_scaler_tier',
        "kind": "scalar",
        "description": "Confidence tier (1=high, 2=medium, 3=low)",
        "output_key": "predictions.tier",
        "unit": "tier",
        "fallback": None,
        "optional": True,
    },
]


# ── End-to-End SLO (from workload generator trace CSV) ──────────────

def compute_e2e_slo_metrics(trace_csv_path: Path, slo_target_ms: float = 200.0) -> dict:
    """Read workload generator trace CSV, compute end-to-end SLO metrics.

    The workload generator saves per-second latency percentiles to
    `workload_*_timeseries.csv` (columns: p50_ms, p95_ms, p99_ms).
    This function extracts true end-to-end latency metrics, which is
    more accurate than the Prometheus frontend-internal histogram
    (which only measures time inside the Flask process, missing
    processor/compute-worker wait time).

    Args:
        trace_csv_path: Path to the workload generator timeseries CSV.
        slo_target_ms: SLO target in milliseconds (default: 200ms).

    Returns:
        dict with:
            e2e_p50_ms: float
            e2e_p95_ms: float
            e2e_p99_ms: float
            e2e_slo_violation_rate: float  # fraction of ticks where p95 > target
            e2e_slo_violation_intervals: int
            e2e_total_intervals: int
    """
    import csv

    trace_csv_path = Path(trace_csv_path)
    if not trace_csv_path.exists():
        logger.warning("Trace CSV not found: %s", trace_csv_path)
        return {
            "e2e_p50_ms": None,
            "e2e_p95_ms": None,
            "e2e_p99_ms": None,
            "e2e_slo_violation_rate": None,
            "e2e_slo_violation_intervals": 0,
            "e2e_total_intervals": 0,
        }

    p50_vals = []
    p95_vals = []
    p99_vals = []
    violations = 0
    total = 0

    with open(trace_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                p50 = float(row.get("p50_ms", 0) or 0)
                p95 = float(row.get("p95_ms", 0) or 0)
                p99 = float(row.get("p99_ms", 0) or 0)
            except (ValueError, KeyError):
                continue
            p50_vals.append(p50)
            p95_vals.append(p95)
            p99_vals.append(p99)
            total += 1
            if p95 > slo_target_ms:
                violations += 1

    if total == 0:
        return {
            "e2e_p50_ms": None,
            "e2e_p95_ms": None,
            "e2e_p99_ms": None,
            "e2e_slo_violation_rate": None,
            "e2e_slo_violation_intervals": 0,
            "e2e_total_intervals": 0,
        }

    return {
        "e2e_p50_ms": round(sum(p50_vals) / total, 1),
        "e2e_p95_ms": round(sorted(p95_vals)[int(total * 0.95)] if total > 1 else p95_vals[0], 1),
        "e2e_p99_ms": round(sorted(p99_vals)[int(total * 0.99)] if total > 1 else p99_vals[0], 1),
        "e2e_slo_violation_rate": round(violations / total, 4),
        "e2e_slo_violation_intervals": violations,
        "e2e_total_intervals": total,
    }
