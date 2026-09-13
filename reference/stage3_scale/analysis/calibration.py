"""Post-hoc calibration analysis for ConfScale controllers.

For each cell that ran a UQ method (confscale-be / confscale-scp / confscale-qr),
we have a `controller_scale_log.json` recording the controller's prediction
interval at every scale decision, and a `timeseries.csv` recording observed
RPS over time. This module joins them to compute empirical coverage relative
to the controller's 90% target (alpha=0.1).

Outputs per cell:
- empirical_coverage: fraction of scale decisions where actual RPS (one
  horizon ahead) fell inside [ci_lower, ci_upper]
- median_width / p95_width: absolute interval width in RPS units
- spike_coverage: coverage restricted to scale decisions where the observed
  RPS was in the top decile for that cell
- width_replica_corr: Spearman correlation between interval width and
  scheduled replicas (signals whether wider intervals drive more replicas)

Per-method aggregates: mean ± std across the 12 cells per method.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

logger = logging.getLogger("calibration")

UQ_METHODS = (
    "confscale-be",
    "confscale-scp",
    "confscale-qr",
    "confscale-scp-online",
    "risk-quantile-scp",
)
HORIZON_STEP = 0           # First-step-ahead horizon (controller scales on this)
TARGET_COVERAGE = 0.90     # 1 - alpha (alpha=0.1 in conformal.py)

# Sub-methods identified from the cell's recal/policy metadata (not the
# directory-name prefix). Offline-SCP cells stay as "confscale-scp"; online
# recalibration shifts them to "confscale-scp-online" so aggregates separate
# the two regimes.
SCP_METHOD_OFFLINE = "confscale-scp"
SCP_METHOD_ONLINE = "confscale-scp-online"


def _iso_to_unix(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _detect_method(cell_dir: Path) -> str | None:
    """Pull method name from the cell directory name (e.g. confscale-scp_b_rep1_...).

    Returns the *base* method (confscale-{be,scp,qr}); use _resolve_method_variant
    after loading the scale log to refine SCP cells into offline vs online-recal.
    """
    parts = cell_dir.name.split("_")
    return parts[0] if parts and parts[0] in UQ_METHODS else None


def _has_recal(scale_df: pd.DataFrame) -> bool:
    """True iff at least one log entry carries a 'recal' block (online-recal SCP)."""
    if scale_df.empty or "recal" not in scale_df.columns:
        return False
    return scale_df["recal"].apply(lambda v: isinstance(v, dict)).any()


def _resolve_method_variant(base_method: str, scale_df: pd.DataFrame) -> str:
    """Refine the base method name using the loaded scale log.

    SCP cells with a recal block become 'confscale-scp-online'; everything else
    keeps the directory-derived label so existing offline-SCP cells continue to
    aggregate together.
    """
    if base_method == SCP_METHOD_OFFLINE and _has_recal(scale_df):
        return SCP_METHOD_ONLINE
    return base_method


def _load_scale_log(cell_dir: Path) -> pd.DataFrame:
    path = cell_dir / "controller_scale_log.json"
    rows = json.loads(path.read_text())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["scale_unix"] = df["timestamp"].apply(_iso_to_unix)
    # Pick out horizon-0 forecast bounds. Some early rows may have empty lists.
    def _pick(col, default=np.nan):
        return df[col].apply(
            lambda v: v[HORIZON_STEP] if isinstance(v, list) and len(v) > HORIZON_STEP else default
        )
    df["ci_lower_h0"] = _pick("ci_lower")
    df["ci_upper_h0"] = _pick("ci_upper")
    df["point_forecast_h0"] = _pick("point_forecast")
    # Online-recal q_hat trajectory: ci_lower/ci_upper above ALREADY reflect
    # the per-iteration q_hat (controller computes them inside the same loop
    # body where q_hat updates land), but the q_hat value itself is useful
    # for diagnostics — extract it where the recal block is present.
    if "recal" in df.columns:
        def _q_hat_h(v):
            if isinstance(v, dict):
                q = v.get("q_hat_used")
                if isinstance(q, list) and len(q) > HORIZON_STEP:
                    return q[HORIZON_STEP]
            return np.nan
        df["q_hat_h0"] = df["recal"].apply(_q_hat_h)
        df["recal_updated"] = df["recal"].apply(
            lambda v: bool(v.get("updated")) if isinstance(v, dict) else False
        )
    else:
        df["q_hat_h0"] = np.nan
        df["recal_updated"] = False
    return df


def _load_operator_summary(cell_dir: Path) -> dict:
    """Read operator_metrics_summary.json if present (online-recal cells expose
    online_recal.initial_q_hat / final_q_hat there)."""
    path = cell_dir / "operator_metrics_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("%s: failed to read operator_metrics_summary.json: %s",
                       cell_dir.name, e)
        return {}


def _load_timeseries(cell_dir: Path) -> pd.DataFrame:
    path = cell_dir / "timeseries.csv"
    ts = pd.read_csv(path)
    if "timestamp" not in ts.columns or "rps" not in ts.columns:
        return pd.DataFrame()
    ts = ts.dropna(subset=["rps", "timestamp"]).copy()
    ts["timestamp"] = ts["timestamp"].astype(float)
    return ts.sort_values("timestamp").reset_index(drop=True)


def _lookup_future_rps(ts: pd.DataFrame, scale_unix: float, horizon_s: float,
                       tolerance_s: float = 30.0) -> float:
    """Return actual RPS at scale_unix + horizon_s, allowing ±tolerance match."""
    if ts.empty:
        return float("nan")
    target = scale_unix + horizon_s
    idx = (ts["timestamp"] - target).abs().idxmin()
    actual_dt = abs(float(ts.loc[idx, "timestamp"]) - target)
    if actual_dt > tolerance_s:
        return float("nan")
    return float(ts.loc[idx, "rps"])


def analyze_cell(cell_dir: Path, interval_s: int = 30) -> dict:
    """Compute calibration stats for a single cell.

    Online-recal-SCP cells (controller_scale_log entries carry a `recal` block)
    are tagged as `confscale-scp-online` and additionally report the q_hat
    trajectory drawn from per-iteration `q_hat_used` and from
    `operator_metrics_summary.json` (`online_recal.initial_q_hat`/`final_q_hat`).
    The empirical-coverage computation is unchanged: ci_lower/ci_upper in the
    log already reflect the q_hat active at each decision, dynamic or not.
    """
    base_method = _detect_method(cell_dir)
    if base_method is None:
        return {}

    scale_df = _load_scale_log(cell_dir)
    ts = _load_timeseries(cell_dir)
    if scale_df.empty or ts.empty:
        logger.warning("%s: empty scale_log or timeseries", cell_dir.name)
        return {}

    method = _resolve_method_variant(base_method, scale_df)
    has_recal = method == SCP_METHOD_ONLINE
    operator_summary = _load_operator_summary(cell_dir) if has_recal else {}

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
        logger.warning("%s: no scale decisions with valid joined RPS", cell_dir.name)
        return {}

    in_interval = (
        (rows["actual_rps_h0"] >= rows["ci_lower_h0"])
        & (rows["actual_rps_h0"] <= rows["ci_upper_h0"])
    )
    widths = rows["ci_upper_h0"] - rows["ci_lower_h0"]

    # Spike: top decile of observed RPS in this cell
    rps_threshold = ts["rps"].quantile(0.90)
    spike_mask = rows["actual_rps_h0"] >= rps_threshold
    spike_cov = in_interval[spike_mask].mean() if spike_mask.any() else float("nan")

    if "actual_replicas" in rows.columns and rows["actual_replicas"].notna().any():
        corr, _ = spearmanr(widths, rows["actual_replicas"], nan_policy="omit")
    else:
        corr = float("nan")

    point_pred = rows["point_forecast_h0"]
    abs_err = (point_pred - rows["actual_rps_h0"]).abs()
    rel_err = abs_err / (rows["actual_rps_h0"].abs() + 1.0)

    # Extract workload letter and replicate number from the cell name
    parts = cell_dir.name.split("_")
    workload = parts[1].upper() if len(parts) > 1 else None
    replicate = parts[2] if len(parts) > 2 else None

    result = {
        "cell": cell_dir.name,
        "method": method,
        "base_method": base_method,
        "online_recal": has_recal,
        "workload": workload,
        "replicate": replicate,
        "n_decisions": int(valid.sum()),
        "empirical_coverage": float(in_interval.mean()),
        "target_coverage": TARGET_COVERAGE,
        "median_width": float(widths.median()),
        "p95_width": float(widths.quantile(0.95)),
        "mean_width": float(widths.mean()),
        "spike_coverage": float(spike_cov) if not np.isnan(spike_cov) else None,
        "spike_threshold_rps": float(rps_threshold),
        "width_replica_corr": float(corr) if not np.isnan(corr) else None,
        "median_abs_err": float(abs_err.median()),
        "mean_rel_err": float(rel_err.mean()),
    }

    if has_recal:
        recal_meta = operator_summary.get("online_recal", {}) if isinstance(
            operator_summary, dict) else {}
        initial_q = recal_meta.get("initial_q_hat")
        final_q = recal_meta.get("final_q_hat")
        q_series_h0 = rows["q_hat_h0"].dropna()
        result.update({
            "initial_q_hat_h0": float(initial_q[HORIZON_STEP])
                if isinstance(initial_q, list) and len(initial_q) > HORIZON_STEP
                else None,
            "final_q_hat_h0": float(final_q[HORIZON_STEP])
                if isinstance(final_q, list) and len(final_q) > HORIZON_STEP
                else None,
            "q_hat_updates": int(recal_meta.get("q_hat_updates", 0))
                if recal_meta else None,
            "q_hat_h0_min": float(q_series_h0.min()) if not q_series_h0.empty else None,
            "q_hat_h0_max": float(q_series_h0.max()) if not q_series_h0.empty else None,
            "q_hat_h0_mean": float(q_series_h0.mean()) if not q_series_h0.empty else None,
            "n_recal_updates_logged": int(rows["recal_updated"].sum()),
        })

    return result


def analyze_all_cells(input_dir: Path, interval_s: int = 30) -> pd.DataFrame:
    cells = sorted(
        p for p in Path(input_dir).iterdir()
        if p.is_dir() and _detect_method(p) is not None
    )
    rows = []
    for cell in cells:
        result = analyze_cell(cell, interval_s=interval_s)
        if result:
            rows.append(result)
    return pd.DataFrame(rows)


def collect_q_hat_timeseries(input_dir: Path) -> pd.DataFrame:
    """Emit a long-form table of q_hat_used over time for online-recal cells.

    Rows: one per logged scale decision in any cell that carries a recal block.
    Columns: cell, method, workload, replicate, elapsed_s, q_hat_h0,
    recal_updated.

    Useful as input to time-series figures or for confirming the calibration
    trajectory (initial → final q_hat) per cell.
    """
    rows = []
    for cell in sorted(p for p in Path(input_dir).iterdir() if p.is_dir()):
        base_method = _detect_method(cell)
        if base_method != SCP_METHOD_OFFLINE:
            continue
        scale_df = _load_scale_log(cell)
        if scale_df.empty or not _has_recal(scale_df):
            continue
        parts = cell.name.split("_")
        workload = parts[1].upper() if len(parts) > 1 else None
        replicate = parts[2] if len(parts) > 2 else None
        for _, r in scale_df.iterrows():
            rows.append({
                "cell": cell.name,
                "method": SCP_METHOD_ONLINE,
                "workload": workload,
                "replicate": replicate,
                "elapsed_s": float(r.get("elapsed_s", np.nan)),
                "q_hat_h0": float(r.get("q_hat_h0"))
                    if pd.notna(r.get("q_hat_h0")) else np.nan,
                "recal_updated": bool(r.get("recal_updated", False)),
            })
    return pd.DataFrame(rows)


def aggregate_by_method(per_cell: pd.DataFrame) -> pd.DataFrame:
    if per_cell.empty:
        return per_cell
    agg = per_cell.groupby("method").agg(
        n_cells=("cell", "count"),
        coverage_mean=("empirical_coverage", "mean"),
        coverage_std=("empirical_coverage", "std"),
        coverage_min=("empirical_coverage", "min"),
        coverage_max=("empirical_coverage", "max"),
        median_width_mean=("median_width", "mean"),
        median_width_std=("median_width", "std"),
        p95_width_mean=("p95_width", "mean"),
        spike_coverage_mean=("spike_coverage", "mean"),
        spike_coverage_std=("spike_coverage", "std"),
        width_replica_corr_mean=("width_replica_corr", "mean"),
        mean_rel_err_mean=("mean_rel_err", "mean"),
    ).reset_index()
    agg["coverage_gap_pct"] = (TARGET_COVERAGE - agg["coverage_mean"]) * 100
    return agg


def aggregate_by_method_workload(per_cell: pd.DataFrame) -> pd.DataFrame:
    if per_cell.empty:
        return per_cell
    return per_cell.groupby(["method", "workload"]).agg(
        n_cells=("cell", "count"),
        coverage_mean=("empirical_coverage", "mean"),
        coverage_std=("empirical_coverage", "std"),
        median_width_mean=("median_width", "mean"),
        spike_coverage_mean=("spike_coverage", "mean"),
    ).reset_index()


METHOD_COLORS = {
    "confscale-be": "#1f77b4",
    "confscale-scp": "#2ca02c",
    "confscale-scp-online": "#17becf",
    "confscale-qr": "#ff7f0e",
}


def write_calibration_figure(per_cell: pd.DataFrame, per_method: pd.DataFrame,
                             output_path: Path) -> None:
    """Two-panel figure: empirical coverage with target, and median-width per workload."""
    if per_cell.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel 1: coverage with 90% target
    methods = per_method["method"].tolist()
    x = np.arange(len(methods))
    colors = [METHOD_COLORS.get(m, "#555555") for m in methods]
    bars = ax1.bar(x, per_method["coverage_mean"], yerr=per_method["coverage_std"],
                   color=colors, alpha=0.85, edgecolor="black", linewidth=0.5,
                   capsize=4)
    ax1.axhline(TARGET_COVERAGE, color="red", linestyle="--", linewidth=1.5,
                label=f"Target ({TARGET_COVERAGE*100:.0f}%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha="right")
    ax1.set_ylabel("Empirical coverage")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Coverage vs target (over all cells)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)
    for bar, cov in zip(bars, per_method["coverage_mean"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{cov*100:.0f}%", ha="center", va="bottom", fontsize=8)

    # Panel 2: per-workload coverage box (12 cells per method, 3 per workload)
    workloads = sorted(per_cell["workload"].dropna().unique())
    positions = []
    box_data = []
    box_colors = []
    labels = []
    for i, m in enumerate(methods):
        for j, w in enumerate(workloads):
            d = per_cell[(per_cell["method"] == m)
                         & (per_cell["workload"] == w)]["empirical_coverage"]
            positions.append(i * (len(workloads) + 1) + j)
            box_data.append(d.values)
            box_colors.append(METHOD_COLORS.get(m, "#555555"))
            labels.append(w)
    bp = ax2.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True,
                     showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(TARGET_COVERAGE, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Empirical coverage")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Coverage by workload (3 cells each)")
    for i, m in enumerate(methods):
        mid = i * (len(workloads) + 1) + (len(workloads) - 1) / 2
        ax2.text(mid, -0.07, m, ha="center", va="top", fontsize=8,
                 color=METHOD_COLORS.get(m), fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Empirical interval coverage vs 90% target (post-hoc)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_markdown_table(per_method: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "### Table 3: UQ method calibration",
        "",
        "| Method | Coverage (target 90%) | Coverage gap | Median width (RPS) | p95 width (RPS) | Spike coverage | Width↔replicas (Spearman) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in per_method.iterrows():
        lines.append(
            f"| {r['method']} "
            f"| {r['coverage_mean']*100:.1f}% ± {r['coverage_std']*100:.1f} "
            f"| {r['coverage_gap_pct']:+.1f} pp "
            f"| {r['median_width_mean']:.2f} "
            f"| {r['p95_width_mean']:.2f} "
            f"| {r['spike_coverage_mean']*100:.1f}% "
            f"| {r['width_replica_corr_mean']:+.2f} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc calibration analysis")
    parser.add_argument("--input", type=Path, required=True,
                        help="Merged cell directory (e.g. outputs/e1_96cell_merged_*)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory for calibration outputs")
    parser.add_argument("--interval", type=int, default=30,
                        help="Controller scale interval in seconds (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning %s for UQ cells", args.input)
    per_cell = analyze_all_cells(args.input, interval_s=args.interval)
    logger.info("Analyzed %d cells", len(per_cell))
    if per_cell.empty:
        logger.error("No UQ cells found in %s", args.input)
        sys.exit(1)

    per_cell.to_csv(data_dir / "calibration_per_cell.csv", index=False)

    by_method = aggregate_by_method(per_cell)
    by_method.to_csv(data_dir / "calibration_per_method.csv", index=False)

    by_method_workload = aggregate_by_method_workload(per_cell)
    by_method_workload.to_csv(
        data_dir / "calibration_per_method_workload.csv", index=False
    )

    # Online-recal trajectory (only writes rows if recal cells are present).
    q_hat_ts = collect_q_hat_timeseries(args.input)
    if not q_hat_ts.empty:
        q_hat_ts.to_csv(data_dir / "online_recal_q_hat_timeseries.csv", index=False)
        logger.info("Wrote q_hat trajectory for %d online-recal rows (%d cells)",
                    len(q_hat_ts), q_hat_ts["cell"].nunique())

    write_markdown_table(by_method, tables_dir / "table_3_calibration.md")

    figures_dir = args.output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_calibration_figure(per_cell, by_method,
                             figures_dir / "fig_8_coverage_calibration.pdf")

    print("\nPer-method calibration:")
    pd.options.display.float_format = "{:.3f}".format
    print(by_method[[
        "method", "n_cells", "coverage_mean", "coverage_std",
        "coverage_gap_pct", "median_width_mean", "spike_coverage_mean",
        "width_replica_corr_mean",
    ]].to_string(index=False))

    print("\nPer-method × workload:")
    print(by_method_workload.to_string(index=False))


if __name__ == "__main__":
    main()
