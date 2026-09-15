#!/usr/bin/env python3
"""λ-sweep analysis for confidence-aware scaling."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
if str(_here.parent) not in sys.path:
    sys.path.insert(0, str(_here.parent))

from analysis.calibration import (
    _load_scale_log,
    _load_timeseries,
    _lookup_future_rps,
    HORIZON_STEP,
    TARGET_COVERAGE,
)

logger = logging.getLogger("lambda_analysis")

LAMBDA_RE = re.compile(r"^confscale-scp-lambda-(?P<lam>[0-9]+(?:\.[0-9]+)?)$")
DEFAULT_SWEEP_GLOB = "p3_lambda_sweep_*"


def _parse_lambda_cell(cell_dir: Path) -> dict | None:
    parts = cell_dir.name.split("_")
    if len(parts) < 3:
        return None
    m = LAMBDA_RE.match(parts[0])
    if not m:
        return None
    workload = parts[1].upper() if len(parts[1]) == 1 else parts[1]
    rep_part = parts[2]
    if not rep_part.startswith("rep"):
        return None
    try:
        replicate = int(rep_part[3:])
    except ValueError:
        return None
    return {
        "method": parts[0],
        "lambda": float(m.group("lam")),
        "workload": workload,
        "replicate": replicate,
    }


def _discover_sweep_cells(sweep_roots: list[Path]) -> list[Path]:
    cells: list[Path] = []
    for root in sweep_roots:
        if not root.exists():
            continue
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and _parse_lambda_cell(entry) is not None:
                cells.append(entry)
    return cells


def _load_metrics_json(cell_dir: Path) -> dict:
    path = cell_dir / "metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("%s: failed to parse metrics.json: %s", cell_dir.name, e)
        return {}


def _compute_calibration(cell_dir: Path, interval_s: int = 30) -> dict:
    scale_df = _load_scale_log(cell_dir)
    ts = _load_timeseries(cell_dir)
    if scale_df.empty or ts.empty:
        return {}

    actuals = scale_df["scale_unix"].apply(
        lambda t: _lookup_future_rps(ts, t, interval_s)
    )
    valid = (
        actuals.notna()
        & scale_df["ci_lower_h0"].notna()
        & scale_df["ci_upper_h0"].notna()
    )
    rows = scale_df.loc[valid].copy()
    rows["actual_rps_h0"] = actuals[valid]
    if rows.empty:
        return {}

    in_interval = (
        (rows["actual_rps_h0"] >= rows["ci_lower_h0"])
        & (rows["actual_rps_h0"] <= rows["ci_upper_h0"])
    )
    widths = rows["ci_upper_h0"] - rows["ci_lower_h0"]
    return {
        "n_decisions": int(valid.sum()),
        "empirical_coverage": float(in_interval.mean()),
        "median_width": float(widths.median()),
        "p95_width": float(widths.quantile(0.95)),
    }


def analyze_cell(cell_dir: Path, interval_s: int = 30) -> dict | None:
    parsed = _parse_lambda_cell(cell_dir)
    if parsed is None:
        return None
    metrics = _load_metrics_json(cell_dir)
    e2e = metrics.get("e2e", {})
    slo = metrics.get("slo", {})
    resources = metrics.get("resources", {})

    cal = _compute_calibration(cell_dir, interval_s=interval_s)

    return {
        "cell": cell_dir.name,
        "method": parsed["method"],
        "lambda": parsed["lambda"],
        "workload": parsed["workload"],
        "replicate": parsed["replicate"],
        "mean_replicas": resources.get("mean_replicas", np.nan),
        "max_replicas": resources.get("max_replicas", np.nan),
        "replica_churn": resources.get("replica_churn", np.nan),
        "slo_violation_rate": e2e.get(
            "slo_violation_rate", slo.get("violation_rate", np.nan)
        ),
        "p50_ms": e2e.get("p50_ms", slo.get("p50_ms", np.nan)),
        "p95_ms": e2e.get("p95_ms", slo.get("p95_ms", np.nan)),
        "p99_ms": e2e.get("p99_ms", slo.get("p99_ms", np.nan)),
        "empirical_coverage": cal.get("empirical_coverage"),
        "median_width": cal.get("median_width"),
        "p95_width": cal.get("p95_width"),
        "n_decisions": cal.get("n_decisions"),
    }


def analyze_all(sweep_roots: list[Path], interval_s: int = 30) -> pd.DataFrame:
    cells = _discover_sweep_cells(sweep_roots)
    if not cells:
        return pd.DataFrame()
    rows = [r for r in (analyze_cell(c, interval_s=interval_s) for c in cells) if r]
    return pd.DataFrame(rows)


def _sem(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return float("nan")
    return float(s.std(ddof=1) / np.sqrt(len(s)))


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Group per-cell metrics by (workload, λ) → mean ± stderr, n_reps."""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    g = df.groupby(["workload", "lambda"])[metric]
    out = g.agg(
        mean="mean",
        stderr=_sem,
        n="count",
        min="min",
        max="max",
    ).reset_index()
    out.rename(columns={
        "mean": f"{metric}_mean",
        "stderr": f"{metric}_stderr",
        "n": f"{metric}_n",
        "min": f"{metric}_min",
        "max": f"{metric}_max",
    }, inplace=True)
    return out.sort_values(["workload", "lambda"]).reset_index(drop=True)


def _pareto_color_for(lam: float) -> tuple:
    cmap = plt.get_cmap("viridis")
    return cmap(float(np.clip(lam, 0.0, 1.0)))


def write_pareto_figure(per_cell: pd.DataFrame, output_path: Path) -> None:
    """Replica savings vs SLO violation rate, one point per (λ, workload, rep) annotated by λ."""
    if per_cell.empty:
        logger.warning("Pareto figure skipped: no λ-sweep cells")
        return
    workloads = sorted(per_cell["workload"].dropna().unique())
    n_w = len(workloads)
    fig, axes = plt.subplots(1, n_w, figsize=(4 * n_w, 4.2), squeeze=False)
    for ax, wl in zip(axes[0], workloads):
        sub = per_cell[per_cell["workload"] == wl]
        for lam, g in sub.groupby("lambda"):
            ax.scatter(
                g["mean_replicas"], g["slo_violation_rate"],
                color=_pareto_color_for(lam), s=60, alpha=0.85,
                edgecolor="black", linewidth=0.4,
                label=f"λ={lam:g}",
            )
        # λ-mean trajectory: mean replicas vs mean SLO per λ
        agg = sub.groupby("lambda").agg(
            mean_replicas=("mean_replicas", "mean"),
            slo=("slo_violation_rate", "mean"),
        ).reset_index().sort_values("lambda")
        ax.plot(agg["mean_replicas"], agg["slo"],
                "-", color="grey", alpha=0.6, linewidth=1.0, zorder=0)
        ax.set_xlabel("Mean replicas")
        ax.set_ylabel("SLO violation rate")
        ax.set_title(f"Workload {wl}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7, frameon=True)
    fig.suptitle("λ-sweep Pareto: replica cost vs SLO violation",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_markdown_summary(per_cell: pd.DataFrame, output_path: Path) -> None:
    """Compact human-readable summary: per-λ averages collapsed across workloads."""
    if per_cell.empty:
        output_path.write_text("# λ-sweep summary\n\n_No λ-sweep cells found._\n")
        return
    by_lambda = per_cell.groupby("lambda").agg(
        n_cells=("cell", "count"),
        workloads=("workload", lambda s: sorted(set(s))),
        mean_replicas=("mean_replicas", "mean"),
        slo_rate=("slo_violation_rate", "mean"),
        p95_ms=("p95_ms", "mean"),
        coverage=("empirical_coverage", "mean"),
    ).reset_index().sort_values("lambda")

    lines = [
        "# λ-sweep summary",
        "",
        f"Total cells analyzed: {len(per_cell)}",
        f"λ values present: {sorted(per_cell['lambda'].unique())}",
        f"Workloads present: {sorted(per_cell['workload'].dropna().unique())}",
        "",
        "## Per-λ aggregates (averaged across all workloads, all reps)",
        "",
        "| λ | n_cells | Workloads | Mean replicas | SLO violation | p95 (ms) | Coverage |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in by_lambda.iterrows():
        wl_str = ",".join(r["workloads"])
        cov = f"{r['coverage']*100:.1f}%" if pd.notna(r["coverage"]) else "—"
        lines.append(
            f"| {r['lambda']:g} | {r['n_cells']} | {wl_str} "
            f"| {r['mean_replicas']:.2f} "
            f"| {r['slo_rate']*100:.2f}% "
            f"| {r['p95_ms']:.0f} "
            f"| {cov} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="λ-sweep replica/SLO/coverage analysis")
    parser.add_argument(
        "--input", type=Path, action="append", default=None,
        help=(f"Sweep root (directory containing per-cell run dirs). "
              f"May be passed multiple times. If omitted, globs "
              f"paper3_experiments/outputs/{DEFAULT_SWEEP_GLOB}."),
    )
    parser.add_argument(
        "--outputs-root", type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs",
        help="Where to look for sweep dirs when --input is not given.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory; creates data/ figures/ tables/ subdirs.",
    )
    parser.add_argument("--interval", type=int, default=30,
                        help="Controller scale interval in seconds (default 30)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.input:
        sweep_roots = list(args.input)
    else:
        sweep_roots = sorted(args.outputs_root.glob(DEFAULT_SWEEP_GLOB))
        logger.info("Globbed %d sweep root(s) under %s",
                    len(sweep_roots), args.outputs_root)

    if not sweep_roots:
        logger.warning("No sweep roots found (looked under %s with glob %s). "
                       "Nothing to do.", args.outputs_root, DEFAULT_SWEEP_GLOB)
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.output / "data"
    fig_dir = args.output / "figures"
    tables_dir = args.output / "tables"
    for d in (data_dir, fig_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    per_cell = analyze_all(sweep_roots, interval_s=args.interval)
    logger.info("Analyzed %d λ-sweep cells across %d sweep root(s)",
                len(per_cell), len(sweep_roots))
    if per_cell.empty:
        logger.warning("No λ-sweep cells found under: %s",
                       ", ".join(str(p) for p in sweep_roots))
        # Still write an empty summary so callers can detect the empty state.
        write_markdown_summary(per_cell, tables_dir / "lambda_summary.md")
        return 0

    per_cell.sort_values(["workload", "lambda", "replicate"], inplace=True)
    per_cell.to_csv(data_dir / "lambda_per_cell.csv", index=False)

    aggregate(per_cell, "mean_replicas").to_csv(
        data_dir / "lambda_vs_replicas.csv", index=False
    )
    aggregate(per_cell, "slo_violation_rate").to_csv(
        data_dir / "lambda_vs_slo.csv", index=False
    )
    aggregate(per_cell, "empirical_coverage").to_csv(
        data_dir / "lambda_vs_coverage.csv", index=False
    )
    aggregate(per_cell, "p95_ms").to_csv(
        data_dir / "lambda_vs_p95_latency.csv", index=False
    )

    write_pareto_figure(per_cell, fig_dir / "fig_lambda_pareto.pdf")
    write_markdown_summary(per_cell, tables_dir / "lambda_summary.md")

    print("\nPer-λ aggregates (mean over all workloads × reps):")
    pd.options.display.float_format = "{:.3f}".format
    print(per_cell.groupby("lambda").agg(
        n_cells=("cell", "count"),
        replicas=("mean_replicas", "mean"),
        slo_rate=("slo_violation_rate", "mean"),
        coverage=("empirical_coverage", "mean"),
        p95_ms=("p95_ms", "mean"),
    ).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
