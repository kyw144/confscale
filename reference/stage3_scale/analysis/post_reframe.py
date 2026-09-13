"""Post-reframe analysis pipeline (Stream C of paper3_agenda_2026-05-25).

Walks the post-reframe batch directories and produces the cost-and-coverage
artifacts for §6 of the post-reframe paper:

    aggregated_metrics_post_reframe.csv  - one row per cell across batches
    tables/table_a_coverage_cost_drift.md
    tables/table_b_ladder_overhead.md
    tables/table_c_resource_overhead_updated.md
    figures/figure_a_cost_vs_coverage.pdf
    statistics/pairwise_raw_vs_laddered.json
    _STATUS.md

Per-cell metric sources (see _COST_EXTRACTION_BRIEF.md §2.1):
    run_config.yaml                 -> method/workload/replicate/duration
    metrics.json.resources          -> overhead_replica_seconds, mean_replicas
    metrics.json.{slo,e2e}          -> p95_ms, slo_violation_rate
    operator_metrics_summary.json   -> coverage_monitor.*, ladder.*

The post-reframe CSV supersedes the five hand-cut JSON extracts in
data/p3_runs/results/ (drift_coverage_extended, aci_laddered_coverage,
pid_laddered_no_drift_60min, rolling_origin_laddered_coverage,
laddered_n5_combined). Those remain on disk as provenance.

Entry point: `python -m stage3_scale.analysis.post_reframe`
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

try:
    from .loader import _parse_run_dirname
except ImportError:  # script-style invocation (`python analysis/post_reframe.py`)
    from analysis.loader import _parse_run_dirname  # type: ignore

logger = logging.getLogger(__name__)


# ── Batch registry ─────────────────────────────────────────────────────────
#
# Each entry maps a batch directory name to a dict of metadata controlling how
# the loader walks it. `subdirs` are sub-paths inside the batch dir that
# themselves contain run directories (e.g. drift_injection's F_phase1/F_phase2/G/H).
# `stream_a_nesting=True` means individual cells may have inner data/p3_runs/...
# clutter from a wrong-cwd orchestrator launch (Stream A bug, brief §2.1.2).

@dataclass
class BatchSpec:
    name: str
    subdirs: list[str] = field(default_factory=lambda: [""])  # "" = walk batch root
    include_in_headline: bool = True  # smoke/alias-variance default off
    stream_a_nesting: bool = False
    note: str = ""


POST_REFRAME_BATCHES: list[BatchSpec] = [
    BatchSpec(
        name="drift_injection_e1_20260523_020148",
        subdirs=["F_phase1", "F_phase2", "G", "H"],
        note="54 cells; 9 hpa-error-monitored lack operator_metrics_summary.json",
    ),
    BatchSpec(name="aci_laddered_validation_20260524"),
    BatchSpec(name="rolling_origin_laddered_validation_20260525"),
    BatchSpec(name="pid_laddered_no_drift_60min_20260525"),
    BatchSpec(
        name="smoke_60min_validation_20260524",
        note="Pattern A 60-min PID and rolling-origin; feeds Table C",
    ),
    BatchSpec(
        name="alias_variance_test_20260524",
        include_in_headline=False,
        note="Variance budget check; excluded from headline tables",
    ),
    BatchSpec(name="laddered_replication_extension_20260525"),
    BatchSpec(
        name="hpa_reactive_drift_20260525",
        stream_a_nesting=True,
        note="Stream A; F/G/H complete (9 cells); no coverage monitor by design",
    ),
    BatchSpec(
        name="hpa_qr_drift_20260525",
        note="hpa-qr-monitored × F/G/H × 3 reps; QR-gauge baseline (miscalibration story)",
    ),
    BatchSpec(
        name="h_divergence_rerun_20260525",
        note="Stream B (hpa-error-monitored gauge-enabled rerun); not launched as of 2026-05-25",
    ),
]


# ── Per-cell extraction ────────────────────────────────────────────────────

def _safe_load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return {}


def _safe_load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def extract_cell(cell_dir: Path, batch: str) -> Optional[dict]:
    """Pull the per-cell metric dict from one run directory.

    Returns None if the cell is unparseable (missing both metrics.json and
    operator_metrics_summary.json). Missing fields are NaN-filled.
    """
    parsed = _parse_run_dirname(cell_dir.name)
    if parsed is None:
        return None

    config = _safe_load_yaml(cell_dir / "run_config.yaml")
    metrics = _safe_load_json(cell_dir / "metrics.json") or {}
    op_summary = _safe_load_json(cell_dir / "operator_metrics_summary.json") or {}

    # If neither metrics.json nor operator_metrics_summary.json exists, the cell is broken
    if not metrics and not op_summary:
        return None

    resources = metrics.get("resources", {}) if isinstance(metrics, dict) else {}
    slo = metrics.get("slo", {}) if isinstance(metrics, dict) else {}
    e2e = metrics.get("e2e", {}) if isinstance(metrics, dict) else {}
    cov = op_summary.get("coverage_monitor") if isinstance(op_summary, dict) else None
    if not isinstance(cov, dict):
        cov = {}
    ladder = op_summary.get("ladder") if isinstance(op_summary, dict) else None
    if not isinstance(ladder, dict):
        ladder = {}

    method_config = config.get("method_config", {}) if isinstance(config, dict) else {}

    duration_s = config.get("duration_s") or 0
    overhead = resources.get("overhead_replica_seconds", np.nan)
    overhead_per_hour = (
        overhead / (duration_s / 3600.0)
        if duration_s and not (isinstance(overhead, float) and np.isnan(overhead))
        else np.nan
    )

    row = {
        # Identity
        "run_id": cell_dir.name,
        "batch": batch,
        "method": parsed["method"],
        "workload": parsed["workload"],
        "replicate": parsed["replicate"],
        "status": config.get("status", "unknown") if isinstance(config, dict) else "unknown",
        "duration_s": duration_s,
        "actual_duration_s": config.get("actual_duration_s", 0) if isinstance(config, dict) else 0,
        # Cost
        "overhead_replica_seconds": overhead,
        "overhead_replica_seconds_per_hour": overhead_per_hour,
        "mean_replicas": resources.get("mean_replicas", np.nan),
        "max_replicas": resources.get("max_replicas", np.nan),
        "replica_churn": resources.get("replica_churn", np.nan),
        "mean_cpu_utilization": resources.get("mean_cpu_utilization", np.nan),
        # SLO (controller view + e2e)
        "p50_ms": slo.get("p50_ms", np.nan),
        "p95_ms": slo.get("p95_ms", np.nan),
        "p99_ms": slo.get("p99_ms", np.nan),
        "slo_violation_rate_controller": slo.get("violation_rate", np.nan),
        "e2e_p50_ms": e2e.get("p50_ms", np.nan),
        "e2e_p95_ms": e2e.get("p95_ms", np.nan),
        "e2e_p99_ms": e2e.get("p99_ms", np.nan),
        "slo_violation_rate": e2e.get("slo_violation_rate", np.nan),
        "slo_violation_intervals": e2e.get("slo_violation_intervals", np.nan),
        "total_intervals": e2e.get("total_intervals", np.nan),
        # Coverage monitor
        "coverage_rate": cov.get("coverage_rate", np.nan),
        "final_alert_state": cov.get("final_alert_state"),
        "total_validated": cov.get("total_validated", np.nan),
        "total_covered": cov.get("total_covered", np.nan),
        # Ladder
        "l0_cycles": ladder.get("l0_cycles", np.nan),
        "l1_cycles": ladder.get("l1_cycles", np.nan),
        "l2_cycles": ladder.get("l2_cycles", np.nan),
        "max_escalation": ladder.get("max_escalation", np.nan),
        "max_level": ladder.get("max_level", np.nan),
        # Controller state
        "final_replicas": op_summary.get("final_replicas", np.nan) if isinstance(op_summary, dict) else np.nan,
        "total_intervals_controller": op_summary.get("total_intervals", np.nan) if isinstance(op_summary, dict) else np.nan,
        "total_scale_operations": op_summary.get("total_scale_operations", np.nan) if isinstance(op_summary, dict) else np.nan,
        "mean_decision_latency_ms": op_summary.get("mean_decision_latency_ms", np.nan) if isinstance(op_summary, dict) else np.nan,
        # Method config knobs
        "ladder_enabled": bool(method_config.get("ladder", False)) if isinstance(method_config, dict) else False,
        "coverage_target": method_config.get("coverage_target", np.nan) if isinstance(method_config, dict) else np.nan,
        "coverage_window": method_config.get("coverage_window", np.nan) if isinstance(method_config, dict) else np.nan,
    }
    return row


# ── Batch discovery ────────────────────────────────────────────────────────

def discover_post_reframe_runs(outputs_root: Path) -> pd.DataFrame:
    """Walk all post-reframe batches and return a one-row-per-cell DataFrame.

    Brief §2.1 — the canonical post-reframe loader. Tolerates missing batches
    (e.g. Stream B not launched), missing operator summaries (hpa-* baselines),
    and the Stream A cwd-nesting bug (cell-level data is still at the cell
    root; the redundant data/p3_runs/... subdir is ignored).
    """
    outputs_root = Path(outputs_root)
    rows: list[dict] = []
    anomalies: list[str] = []

    for spec in POST_REFRAME_BATCHES:
        batch_dir = outputs_root / spec.name
        if not batch_dir.exists():
            anomalies.append(f"batch_missing: {spec.name}")
            continue

        for sub in spec.subdirs:
            walk_root = batch_dir / sub if sub else batch_dir
            if not walk_root.exists():
                anomalies.append(f"subdir_missing: {spec.name}/{sub}")
                continue
            for entry in sorted(walk_root.iterdir()):
                if not entry.is_dir():
                    continue
                if _parse_run_dirname(entry.name) is None:
                    continue
                row = extract_cell(entry, batch=spec.name)
                if row is None:
                    anomalies.append(f"unparseable_cell: {spec.name}/{sub}/{entry.name}")
                    continue
                row["batch_subdir"] = sub
                row["include_in_headline"] = spec.include_in_headline
                rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        logger.warning("Discovered zero post-reframe cells under %s", outputs_root)
    else:
        logger.info(
            "Discovered %d post-reframe cells across %d batches",
            len(df), df["batch"].nunique(),
        )
    df.attrs["anomalies"] = anomalies
    return df


# ── Pooling and aggregation helpers ────────────────────────────────────────

# Methods whose laddered n=5 pool comes from drift_injection_e1 (n=3) + laddered_replication_extension (n=2)
LADDERED_POOL_SOURCES: dict[str, list[str]] = {
    "confscale-pid-laddered": ["drift_injection_e1_20260523_020148", "laddered_replication_extension_20260525"],
    "confscale-aci-laddered": ["aci_laddered_validation_20260524", "laddered_replication_extension_20260525"],
    "confscale-rolling-origin-laddered": ["rolling_origin_laddered_validation_20260525"],
}

# Raw methods whose n=3 baseline is in drift_injection_e1
RAW_BASELINE_BATCH: dict[str, str] = {
    "confscale-pid": "drift_injection_e1_20260523_020148",
    "confscale-aci": "drift_injection_e1_20260523_020148",
    "confscale-rolling-origin": "drift_injection_e1_20260523_020148",
}

DRIFT_PATTERNS = ["F", "G", "H"]
COVERAGE_FLAG_THRESHOLD = 0.85


def select_pool(df: pd.DataFrame, method: str, pattern: str) -> pd.DataFrame:
    """Pull the per-rep pool for (method, pattern) per the agenda §3.2 rule."""
    if method in LADDERED_POOL_SOURCES:
        batches = LADDERED_POOL_SOURCES[method]
    elif method in RAW_BASELINE_BATCH:
        batches = [RAW_BASELINE_BATCH[method]]
    else:
        batches = list(df["batch"].unique())
    return df[
        (df["method"] == method)
        & (df["workload"] == pattern)
        & (df["batch"].isin(batches))
    ].copy()


def _mean_std_n(series: pd.Series) -> tuple[float, float, int]:
    vals = series.dropna().values
    if len(vals) == 0:
        return (np.nan, np.nan, 0)
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals))


# ── Table A: coverage + cost on F/G/H ──────────────────────────────────────

TABLE_A_METHODS = [
    "confscale-pid",
    "confscale-pid-laddered",
    "confscale-aci",
    "confscale-aci-laddered",
    "confscale-rolling-origin",
    "confscale-rolling-origin-laddered",
    "hpa-error-monitored",
    "hpa-qr-monitored",
    "hpa-reactive",
]


def build_table_a(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = []
    for method in TABLE_A_METHODS:
        for pat in DRIFT_PATTERNS:
            pool = select_pool(df, method, pat)
            cov_m, cov_s, cov_n = _mean_std_n(pool["coverage_rate"])
            cost_m, cost_s, cost_n = _mean_std_n(pool["overhead_replica_seconds_per_hour"])
            flag = ""
            if not np.isnan(cov_m) and cov_m < COVERAGE_FLAG_THRESHOLD:
                flag = " ⚠️"
            rows.append({
                "method": method,
                "pattern": pat,
                "coverage_mean": cov_m,
                "coverage_std": cov_s,
                "coverage_n": cov_n,
                "cost_mean": cost_m,
                "cost_std": cost_s,
                "cost_n": cost_n,
                "flag": flag,
            })
    out = pd.DataFrame(rows)

    # Markdown rendering
    md = ["# Table A — Coverage and cost on F/G/H",
          "",
          "Per-cell `coverage_rate` from `operator_metrics_summary.json.coverage_monitor`,",
          "and `overhead_replica_seconds_per_hour = overhead_replica_seconds / (duration_s/3600)`",
          "from `metrics.json.resources`. Laddered methods pool n=5 (drift batch + replication extension);",
          "raw methods n=3 (drift batch). Coverage < 0.85 flagged ⚠️ per agenda §3.2.",
          "",
          "| method | pattern | coverage (mean ± std) | overhead_rs/h (mean ± std) | n (cov/cost) |",
          "|---|---|---|---|---|"]
    for r in rows:
        cov = "—" if r["coverage_n"] == 0 else f"{r['coverage_mean']:.4f} ± {r['coverage_std']:.4f}{r['flag']}"
        cost = "—" if r["cost_n"] == 0 else f"{r['cost_mean']:.1f} ± {r['cost_std']:.1f}"
        md.append(f"| `{r['method']}` | {r['pattern']} | {cov} | {cost} | {r['coverage_n']}/{r['cost_n']} |")

    md.extend([
        "",
        "## Notes",
        "- `hpa-reactive` has no coverage monitor by design; cost-only.",
        "- `hpa-error-monitored` coverage is NaN until Stream B (gauge-enabled rerun) lands. Cost rows populated from the drift batch.",
        "- Stream A produced Pattern F only — `hpa-reactive` G and H rows are empty pending additional runs.",
    ])
    return out, "\n".join(md)


# ── Table B: raw vs laddered delta ─────────────────────────────────────────

LADDER_PAIRS = [
    ("confscale-pid", "confscale-pid-laddered"),
    ("confscale-aci", "confscale-aci-laddered"),
    ("confscale-rolling-origin", "confscale-rolling-origin-laddered"),
]


def build_table_b(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = []
    for raw, lad in LADDER_PAIRS:
        for pat in DRIFT_PATTERNS:
            raw_pool = select_pool(df, raw, pat)
            lad_pool = select_pool(df, lad, pat)
            raw_cov, _, raw_n = _mean_std_n(raw_pool["coverage_rate"])
            lad_cov, _, lad_n = _mean_std_n(lad_pool["coverage_rate"])
            raw_cost, _, _ = _mean_std_n(raw_pool["overhead_replica_seconds_per_hour"])
            lad_cost, _, _ = _mean_std_n(lad_pool["overhead_replica_seconds_per_hour"])
            d_cov_pp = (lad_cov - raw_cov) * 100 if not (np.isnan(raw_cov) or np.isnan(lad_cov)) else np.nan
            d_cost_pct = ((lad_cost - raw_cost) / raw_cost) * 100 if raw_cost and not np.isnan(raw_cost) and not np.isnan(lad_cost) else np.nan
            rows.append({
                "inner_loop": raw.replace("confscale-", ""),
                "pattern": pat,
                "raw_coverage": raw_cov,
                "ladder_coverage": lad_cov,
                "delta_coverage_pp": d_cov_pp,
                "raw_cost_rs_per_hour": raw_cost,
                "ladder_cost_rs_per_hour": lad_cost,
                "delta_cost_pct": d_cost_pct,
                "raw_n": raw_n,
                "ladder_n": lad_n,
            })
    out = pd.DataFrame(rows)

    md = ["# Table B — Ladder overhead delta (raw → laddered)",
          "",
          "ΔCoverage in percentage points, ΔCost in percent change. Raw n=3 (drift batch);",
          "laddered n=5 (drift batch + replication extension) where pooling applies.",
          "Cells with pooled mean coverage < 0.85 are noted in Table A.",
          "",
          "| inner loop | pattern | raw cov | ladder cov | ΔCov (pp) | raw cost/h | ladder cost/h | ΔCost (%) | raw n | ladder n |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def fmt(v, p=4):
            return "—" if np.isnan(v) else f"{v:.{p}f}"
        md.append(
            f"| {r['inner_loop']} | {r['pattern']} | "
            f"{fmt(r['raw_coverage'])} | {fmt(r['ladder_coverage'])} | "
            f"{fmt(r['delta_coverage_pp'], 2)} | "
            f"{fmt(r['raw_cost_rs_per_hour'], 1)} | {fmt(r['ladder_cost_rs_per_hour'], 1)} | "
            f"{fmt(r['delta_cost_pct'], 2)} | {r['raw_n']} | {r['ladder_n']} |"
        )
    return out, "\n".join(md)


# ── Table C: Pattern A 60-min PID rows added to the original Table 2 ───────

def build_table_c(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Append confscale-pid and confscale-pid-laddered Pattern A 60-min rows.

    The original Table 2 (resource overhead on A/B/D) is at
    data/p3_runs/results/real/tables/table_2_resource_overhead.md.
    This output mirrors the structure with the new rows added and flags the
    B/D gap per brief §2.4.
    """
    a_pid = df[
        (df["method"] == "confscale-pid")
        & (df["workload"] == "A")
        & (df["batch"] == "smoke_60min_validation_20260524")
    ]
    a_pid_lad = df[
        (df["method"] == "confscale-pid-laddered")
        & (df["workload"] == "A")
        & (df["batch"] == "pid_laddered_no_drift_60min_20260525")
    ]

    rows = []
    for label, pool in [("confscale-pid", a_pid), ("confscale-pid-laddered", a_pid_lad)]:
        cost_m, cost_s, cost_n = _mean_std_n(pool["overhead_replica_seconds_per_hour"])
        repl_m, repl_s, _ = _mean_std_n(pool["mean_replicas"])
        cov_m, cov_s, cov_n = _mean_std_n(pool["coverage_rate"])
        rows.append({
            "method": label,
            "pattern": "A (60-min)",
            "overhead_rs_per_hour_mean": cost_m,
            "overhead_rs_per_hour_std": cost_s,
            "mean_replicas_mean": repl_m,
            "mean_replicas_std": repl_s,
            "coverage_mean": cov_m,
            "coverage_std": cov_s,
            "n": cost_n,
        })
    out = pd.DataFrame(rows)

    md = ["# Table C — Resource overhead with laddered methods (Pattern A 60-min)",
          "",
          "Appends confscale-pid and confscale-pid-laddered Pattern A 60-min rows to the",
          "structure of `results/real/tables/table_2_resource_overhead.md` (pre-reframe baseline).",
          "B and D are intentionally empty: no laddered runs exist for those patterns at 60-min.",
          "Closing that gap is an optional follow-up per agenda §7.",
          "",
          "| method | pattern | overhead_rs/h (mean ± std) | mean_replicas (mean ± std) | coverage (mean ± std) | n |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        cov = "—" if r["n"] == 0 or np.isnan(r["coverage_mean"]) else f"{r['coverage_mean']:.4f} ± {r['coverage_std']:.4f}"
        md.append(
            f"| `{r['method']}` | {r['pattern']} | "
            f"{r['overhead_rs_per_hour_mean']:.1f} ± {r['overhead_rs_per_hour_std']:.1f} | "
            f"{r['mean_replicas_mean']:.2f} ± {r['mean_replicas_std']:.2f} | "
            f"{cov} | {r['n']} |"
        )
    md.extend([
        "",
        "## Gap",
        "- `confscale-pid-laddered × {B, D} × 60-min` not run. Agenda §7 lists this as a deferred follow-up.",
    ])
    return out, "\n".join(md)


# ── Pairwise stats ─────────────────────────────────────────────────────────

def run_pairwise_tests(df: pd.DataFrame) -> dict:
    """Per agenda §3.3: Welch's t-test + Cohen's d, Holm-Bonferroni across 9 comparisons.

    Imports the helpers from stats.py.
    """
    try:
        from .stats import welch_t_test, cohens_d_independent, bonferroni_holm_correction
    except ImportError:
        from analysis.stats import welch_t_test, cohens_d_independent, bonferroni_holm_correction  # type: ignore

    comparisons: list[dict] = []
    cov_pvals: list[float] = []
    cost_pvals: list[float] = []
    for raw, lad in LADDER_PAIRS:
        for pat in DRIFT_PATTERNS:
            raw_pool = select_pool(df, raw, pat)
            lad_pool = select_pool(df, lad, pat)
            raw_cov = raw_pool["coverage_rate"].dropna().values
            lad_cov = lad_pool["coverage_rate"].dropna().values
            raw_cost = raw_pool["overhead_replica_seconds_per_hour"].dropna().values
            lad_cost = lad_pool["overhead_replica_seconds_per_hour"].dropna().values

            cov_test = welch_t_test(lad_cov, raw_cov)
            cost_test = welch_t_test(lad_cost, raw_cost)
            cov_d = cohens_d_independent(lad_cov, raw_cov)
            cost_d = cohens_d_independent(lad_cost, raw_cost)

            cov_pvals.append(cov_test.get("p_value", np.nan))
            cost_pvals.append(cost_test.get("p_value", np.nan))

            comparisons.append({
                "comparison": f"{lad} vs {raw}",
                "pattern": pat,
                "raw_n": len(raw_cov),
                "ladder_n": len(lad_cov),
                "coverage": {
                    "raw_mean": float(np.mean(raw_cov)) if len(raw_cov) else None,
                    "ladder_mean": float(np.mean(lad_cov)) if len(lad_cov) else None,
                    **cov_test,
                    "cohens_d": cov_d,
                },
                "cost": {
                    "raw_mean": float(np.mean(raw_cost)) if len(raw_cost) else None,
                    "ladder_mean": float(np.mean(lad_cost)) if len(lad_cost) else None,
                    **cost_test,
                    "cohens_d": cost_d,
                },
            })

    # Holm-Bonferroni separately on coverage and cost p-value sets
    cov_corr = bonferroni_holm_correction([p if not np.isnan(p) else 1.0 for p in cov_pvals])
    cost_corr = bonferroni_holm_correction([p if not np.isnan(p) else 1.0 for p in cost_pvals])
    for c, pc_cov, pc_cost in zip(comparisons, cov_corr, cost_corr):
        c["coverage"]["p_value_holm"] = pc_cov
        c["coverage"]["significant_holm"] = pc_cov < 0.05
        c["cost"]["p_value_holm"] = pc_cost
        c["cost"]["significant_holm"] = pc_cost < 0.05

    return {
        "n_comparisons": len(comparisons),
        "test": "Welch's t-test (independent samples, unequal variance)",
        "effect_size": "Cohen's d (independent, pooled SD)",
        "correction": "Holm-Bonferroni (applied separately to coverage and cost p-value sets)",
        "comparisons": comparisons,
    }


# ── Figure A: cost vs coverage scatter ─────────────────────────────────────

def make_figure_a(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    methods = [
        ("confscale-pid",                  "o", "C0", "raw"),
        ("confscale-pid-laddered",         "s", "C0", "ladder"),
        ("confscale-aci",                  "o", "C1", "raw"),
        ("confscale-aci-laddered",         "s", "C1", "ladder"),
        ("confscale-rolling-origin",       "o", "C2", "raw"),
        ("confscale-rolling-origin-laddered","s","C2", "ladder"),
        ("hpa-error-monitored",            "^", "C3", "baseline"),
        ("hpa-qr-monitored",               "D", "C5", "baseline"),
        ("hpa-reactive",                   "v", "C4", "baseline"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, pat in zip(axes, DRIFT_PATTERNS):
        for method, marker, color, _kind in methods:
            pool = select_pool(df, method, pat)
            if len(pool) == 0:
                continue
            x = pool["coverage_rate"].values
            y = pool["overhead_replica_seconds_per_hour"].values
            ax.scatter(x, y, marker=marker, c=color, label=method, s=42,
                       edgecolors="black", linewidths=0.4, alpha=0.85)
        ax.axvline(0.85, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(0.90, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_yscale("log")
        ax.set_xlabel("Coverage rate")
        ax.set_title(f"Pattern {pat}")
        ax.set_xlim(0.0, 1.0)
    axes[0].set_ylabel("Overhead replica-seconds / hour (log)")
    # One legend on the right
    handles, labels = axes[-1].get_legend_handles_labels()
    seen: set[str] = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if uniq:
        fig.legend([h for h, _ in uniq], [l for _, l in uniq],
                   loc="center right", bbox_to_anchor=(1.13, 0.5), fontsize=8, frameon=False)
    fig.suptitle("Cost vs coverage on drift patterns (vertical lines: 0.85 escalate, 0.90 target)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote figure %s", output_path)


# ── Provenance: compare to existing JSON extracts ──────────────────────────

PROVENANCE_EXTRACTS = [
    "drift_coverage_extended_20260524.json",
    "aci_laddered_coverage_20260524.json",
    "pid_laddered_no_drift_60min_20260525.json",
    "rolling_origin_laddered_coverage_20260525.json",
    "laddered_n5_combined_20260525.json",
]


def cross_check_against_extracts(df: pd.DataFrame, results_root: Path) -> list[dict]:
    """Sanity-check the CSV-derived n=5 means against laddered_n5_combined_20260525.json."""
    extract_path = results_root / "laddered_n5_combined_20260525.json"
    if not extract_path.exists():
        return [{"warning": f"missing extract: {extract_path}"}]
    extract = _safe_load_json(extract_path) or {}
    map_to_method = {
        "aci_laddered_n5": "confscale-aci-laddered",
        "pid_laddered_n5": "confscale-pid-laddered",
    }
    diffs = []
    for key, method in map_to_method.items():
        block = extract.get(key, {})
        for pat, ref in block.items():
            pool = select_pool(df, method, pat)
            csv_mean, _, csv_n = _mean_std_n(pool["coverage_rate"])
            ref_mean = ref.get("mean")
            ref_n = ref.get("n")
            d = {
                "method": method,
                "pattern": pat,
                "csv_n": csv_n,
                "extract_n": ref_n,
                "csv_mean": round(csv_mean, 4) if not np.isnan(csv_mean) else None,
                "extract_mean": ref_mean,
                "delta": round(csv_mean - ref_mean, 4) if (ref_mean is not None and not np.isnan(csv_mean)) else None,
            }
            diffs.append(d)
    return diffs


# ── Orchestrator ───────────────────────────────────────────────────────────

def run_pipeline(outputs_root: Path, results_root: Path) -> dict:
    out_dir = results_root / "post_reframe"
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "statistics").mkdir(parents=True, exist_ok=True)

    df = discover_post_reframe_runs(outputs_root)
    if len(df) == 0:
        raise RuntimeError(f"No post-reframe cells discovered under {outputs_root}")

    # Unified CSV
    csv_path = out_dir / "aggregated_metrics_post_reframe.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", csv_path, len(df))

    # Tables
    _table_a_df, table_a_md = build_table_a(df)
    (out_dir / "tables" / "table_a_coverage_cost_drift.md").write_text(table_a_md + "\n")
    _table_b_df, table_b_md = build_table_b(df)
    (out_dir / "tables" / "table_b_ladder_overhead.md").write_text(table_b_md + "\n")
    _table_c_df, table_c_md = build_table_c(df)
    (out_dir / "tables" / "table_c_resource_overhead_updated.md").write_text(table_c_md + "\n")

    # Stats
    stats_result = run_pairwise_tests(df)
    (out_dir / "statistics" / "pairwise_raw_vs_laddered.json").write_text(
        json.dumps(stats_result, indent=2, default=_json_default)
    )

    # Figure
    make_figure_a(df, out_dir / "figures" / "figure_a_cost_vs_coverage.pdf")

    # Provenance cross-check
    cross = cross_check_against_extracts(df, results_root)
    (out_dir / "statistics" / "provenance_cross_check.json").write_text(
        json.dumps(cross, indent=2, default=_json_default)
    )

    return {
        "n_cells": len(df),
        "n_anomalies": len(df.attrs.get("anomalies", [])),
        "anomalies": df.attrs.get("anomalies", []),
        "csv_path": str(csv_path),
        "table_paths": [str(p) for p in (out_dir / "tables").glob("*.md")],
        "figure_path": str(out_dir / "figures" / "figure_a_cost_vs_coverage.pdf"),
        "stats_path": str(out_dir / "statistics" / "pairwise_raw_vs_laddered.json"),
        "provenance_cross_check": cross,
        "stats": stats_result,
    }


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o) if not np.isnan(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"unserializable: {type(o)}")


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    ap = argparse.ArgumentParser(description="Stream C — post-reframe cost extraction pipeline")
    ap.add_argument("--outputs", default="data/p3_runs/outputs",
                    help="Path to data/p3_runs/outputs/")
    ap.add_argument("--results", default="data/p3_runs/results",
                    help="Path to data/p3_runs/results/")
    args = ap.parse_args()
    result = run_pipeline(Path(args.outputs), Path(args.results))
    print(json.dumps({k: v for k, v in result.items() if k != "stats"}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
