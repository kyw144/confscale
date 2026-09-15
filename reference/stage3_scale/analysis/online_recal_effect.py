#!/usr/bin/env python3
"""Offline-SCP vs online-recal-SCP comparison."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
if str(_here.parent) not in sys.path:
    sys.path.insert(0, str(_here.parent))

from analysis.calibration import (
    analyze_all_cells,
    TARGET_COVERAGE,
    SCP_METHOD_OFFLINE,
    SCP_METHOD_ONLINE,
)
from analysis.loader import load_runs

logger = logging.getLogger("online_recal_effect")

CLOSURE_THRESHOLD = 0.60  # ≥ this fraction of the coverage gap closed


def _collect_scp_cells(inputs: list[Path], expected_method: str,
                       interval_s: int = 30) -> pd.DataFrame:
    frames = []
    for inp in inputs:
        if not inp.exists():
            logger.warning("Input dir does not exist: %s", inp)
            continue
        df = analyze_all_cells(inp, interval_s=interval_s)
        if df.empty:
            continue
        df = df.assign(source_dir=str(inp))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_cells = pd.concat(frames, ignore_index=True)
    return all_cells[all_cells["method"] == expected_method].reset_index(drop=True)


def _collect_run_metrics(inputs: list[Path]) -> pd.DataFrame:
    frames = []
    for inp in inputs:
        if not inp.exists():
            continue
        try:
            df = load_runs(inp)
        except Exception as e:
            logger.warning("load_runs(%s) failed: %s", inp, e)
            continue
        if df.empty:
            continue
        frames.append(df.assign(source_dir=str(inp)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _merge_with_metrics(per_cell: pd.DataFrame,
                        metrics_df: pd.DataFrame) -> pd.DataFrame:
    if per_cell.empty:
        return per_cell
    cols = ["run_id", "mean_replicas", "slo_violation_rate", "p95_ms",
            "p99_ms", "replica_churn"]
    if metrics_df.empty:
        for c in cols[1:]:
            per_cell[c] = np.nan
        return per_cell
    keep = metrics_df[[c for c in cols if c in metrics_df.columns]].copy()
    keep.rename(columns={"run_id": "cell"}, inplace=True)
    return per_cell.merge(keep, on="cell", how="left")


def _sem(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return float("nan")
    return float(s.std(ddof=1) / np.sqrt(len(s)))


def _aggregate_workload(df: pd.DataFrame, method_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    keep_metrics = [m for m in
                    ("empirical_coverage", "median_width", "mean_replicas",
                     "slo_violation_rate", "p95_ms")
                    if m in df.columns]
    agg = df.groupby("workload")[keep_metrics].agg(["mean", _sem, "count"])
    agg.columns = [f"{label}_{stat}" for label, stat in agg.columns]
    agg = agg.reset_index()
    agg.insert(0, "method", method_label)
    return agg


def _coverage_closure(off: float, on: float, target: float = TARGET_COVERAGE) -> float:
    gap = target - off
    if not np.isfinite(off) or not np.isfinite(on):
        return float("nan")
    if gap <= 0:
        return float("nan")
    return float((on - off) / gap)


def build_comparison_table(off_cells: pd.DataFrame,
                           on_cells: pd.DataFrame) -> pd.DataFrame:
    """Per-workload row: offline_cov, online_cov, closure %, replica delta, p95 delta."""
    if off_cells.empty and on_cells.empty:
        return pd.DataFrame()

    off_agg = _aggregate_workload(off_cells, "offline")
    on_agg = _aggregate_workload(on_cells, "online")

    workloads = sorted(
        set(off_agg.get("workload", pd.Series(dtype=str)).tolist())
        | set(on_agg.get("workload", pd.Series(dtype=str)).tolist())
    )

    rows = []
    for wl in workloads:
        off_row = off_agg[off_agg["workload"] == wl] if not off_agg.empty else pd.DataFrame()
        on_row = on_agg[on_agg["workload"] == wl] if not on_agg.empty else pd.DataFrame()

        def _val(row, col):
            if row.empty or col not in row.columns:
                return float("nan")
            v = row[col].iloc[0]
            return float(v) if v is not None and pd.notna(v) else float("nan")

        off_cov = _val(off_row, "empirical_coverage_mean")
        on_cov = _val(on_row, "empirical_coverage_mean")
        off_rep = _val(off_row, "mean_replicas_mean")
        on_rep = _val(on_row, "mean_replicas_mean")
        off_p95 = _val(off_row, "p95_ms_mean")
        on_p95 = _val(on_row, "p95_ms_mean")
        off_slo = _val(off_row, "slo_violation_rate_mean")
        on_slo = _val(on_row, "slo_violation_rate_mean")

        def _count(row, col):
            v = _val(row, col)
            return int(v) if np.isfinite(v) else 0

        rows.append({
            "workload": wl,
            "n_offline": _count(off_row, "empirical_coverage_count"),
            "n_online": _count(on_row, "empirical_coverage_count"),
            "offline_coverage": off_cov,
            "online_coverage": on_cov,
            "coverage_delta": on_cov - off_cov if np.isfinite(on_cov) and np.isfinite(off_cov) else float("nan"),
            "coverage_gap_closure": _coverage_closure(off_cov, on_cov),
            "offline_mean_replicas": off_rep,
            "online_mean_replicas": on_rep,
            "replica_delta": on_rep - off_rep if np.isfinite(on_rep) and np.isfinite(off_rep) else float("nan"),
            "offline_p95_ms": off_p95,
            "online_p95_ms": on_p95,
            "p95_delta_ms": on_p95 - off_p95 if np.isfinite(on_p95) and np.isfinite(off_p95) else float("nan"),
            "offline_slo_rate": off_slo,
            "online_slo_rate": on_slo,
        })

    return pd.DataFrame(rows)


def overall_closure(comparison: pd.DataFrame) -> float:
    """Mean coverage-gap closure across workloads where both sides exist."""
    if comparison.empty or "coverage_gap_closure" not in comparison.columns:
        return float("nan")
    valid = comparison["coverage_gap_closure"].dropna()
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def narrative_verdict(closure: float) -> dict:
    """Map a closure fraction to the open-Q7 framing recommendation."""
    if not np.isfinite(closure):
        return {
            "closure": closure,
            "framing": "indeterminate",
            "rationale": "Online or offline data missing — cannot compute closure.",
        }
    if closure >= CLOSURE_THRESHOLD:
        return {
            "closure": closure,
            "framing": "propose-AND-validate",
            "rationale": (
                f"Online recalibration closes {closure*100:.1f}% of the offline "
                f"SCP coverage gap (threshold {CLOSURE_THRESHOLD*100:.0f}%). "
                "We propose split-conformal *and* demonstrate the streaming-"
                "recal fix reaches usable coverage in practice."
            ),
        }
    return {
        "closure": closure,
        "framing": "own-the-drift",
        "rationale": (
            f"Online recalibration closes only {closure*100:.1f}% of the gap "
            f"(< {CLOSURE_THRESHOLD*100:.0f}% threshold). Frame the paper "
            "around honest reporting of the exchangeability violation rather "
            "than a closure claim."
        ),
    }


def write_coverage_figure(comparison: pd.DataFrame, output_path: Path) -> None:
    """Side-by-side bars: offline coverage vs online coverage per workload, with the 90% target line and closure-percentage annotations."""
    if comparison.empty:
        logger.warning("Coverage figure skipped: empty comparison table.")
        return
    valid = comparison.dropna(subset=["offline_coverage", "online_coverage"], how="all")
    if valid.empty:
        logger.warning("Coverage figure skipped: no workloads with any coverage data.")
        return

    workloads = valid["workload"].tolist()
    x = np.arange(len(workloads))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(5.5, len(workloads) * 1.4), 4.0))
    off_vals = valid["offline_coverage"].fillna(0).values
    on_vals = valid["online_coverage"].fillna(0).values
    ax.bar(x - w / 2, off_vals, w, label="offline SCP",
           color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, on_vals, w, label="online-recal SCP",
           color="#17becf", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.axhline(TARGET_COVERAGE, color="red", linestyle="--", linewidth=1.5,
               label=f"Target ({TARGET_COVERAGE*100:.0f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Offline vs online-recal SCP coverage (per workload)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    for xi, (_, row) in zip(x, valid.iterrows()):
        c = row["coverage_gap_closure"]
        if np.isfinite(c):
            ax.text(xi, max(off_vals[xi - 0] if False else 0,
                            row.get("online_coverage", 0) or 0) + 0.04,
                    f"Δ={c*100:+.0f}% gap",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_markdown_summary(comparison: pd.DataFrame, verdict: dict,
                           output_path: Path) -> None:
    lines = [
        "# Online-recal SCP vs offline SCP",
        "",
        f"Target coverage: **{TARGET_COVERAGE*100:.0f}%**",
        f"Closure threshold: **{CLOSURE_THRESHOLD*100:.0f}%** of the gap",
        "",
        "## Verdict",
        "",
        f"- **Framing:** `{verdict['framing']}`",
        f"- **Overall coverage-gap closure:** "
        f"{verdict['closure']*100:.1f}%" if np.isfinite(verdict['closure'])
        else f"- **Overall coverage-gap closure:** undefined",
        f"- **Rationale:** {verdict['rationale']}",
        "",
        "## Per-workload comparison",
        "",
        "| WL | n off | n on | Offline cov | Online cov | Closure | Replicas Δ | p95 Δ (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if comparison.empty:
        lines.append("| — | — | — | — | — | — | — | — |")
    else:
        for _, r in comparison.iterrows():
            def _pct(v):
                return f"{v*100:.1f}%" if np.isfinite(v) else "—"
            def _num(v, fmt="{:+.2f}"):
                return fmt.format(v) if np.isfinite(v) else "—"
            lines.append(
                f"| {r['workload']} "
                f"| {int(r['n_offline'])} | {int(r['n_online'])} "
                f"| {_pct(r['offline_coverage'])} "
                f"| {_pct(r['online_coverage'])} "
                f"| {_pct(r['coverage_gap_closure'])} "
                f"| {_num(r['replica_delta'])} "
                f"| {_num(r['p95_delta_ms'], '{:+.1f}')} |"
            )
    output_path.write_text("\n".join(lines) + "\n")


def _default_online_globs(outputs_root: Path) -> list[Path]:
    return sorted(outputs_root.glob("p3_uq_baselines_*"))


def _default_offline_globs(outputs_root: Path) -> list[Path]:
    matches = sorted(outputs_root.glob("e1_96cell_merged_*"))
    if matches:
        return matches
    return sorted(outputs_root.glob("e1_full_96_*"))


def main():
    parser = argparse.ArgumentParser(
        description="Offline-SCP vs online-recal-SCP comparison (open-Q7 verdict)",
    )
    parser.add_argument(
        "--offline-input", type=Path, action="append", default=None,
        help=("Run dir(s) holding offline-SCP cells. May be repeated. "
              "Default: outputs/e1_96cell_merged_*."),
    )
    parser.add_argument(
        "--online-input", type=Path, action="append", default=None,
        help=("Run dir(s) holding online-recal-SCP cells. May be repeated. "
              "Default: outputs/p3_uq_baselines_*."),
    )
    parser.add_argument(
        "--outputs-root", type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs",
        help="Where to look for the default offline/online globs.",
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

    offline_inputs = args.offline_input or _default_offline_globs(args.outputs_root)
    online_inputs = args.online_input or _default_online_globs(args.outputs_root)
    logger.info("Offline inputs: %s",
                ", ".join(str(p) for p in offline_inputs) or "(none)")
    logger.info("Online inputs:  %s",
                ", ".join(str(p) for p in online_inputs) or "(none)")

    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.output / "data"
    fig_dir = args.output / "figures"
    tables_dir = args.output / "tables"
    for d in (data_dir, fig_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    off_cells = _collect_scp_cells(offline_inputs, SCP_METHOD_OFFLINE,
                                   interval_s=args.interval)
    on_cells = _collect_scp_cells(online_inputs, SCP_METHOD_ONLINE,
                                  interval_s=args.interval)
    logger.info("Loaded %d offline-SCP cells, %d online-recal-SCP cells",
                len(off_cells), len(on_cells))

    if off_cells.empty and on_cells.empty:
        logger.warning("No SCP cells found on either side; nothing to write.")
        verdict = {
            "closure": float("nan"),
            "framing": "no-data",
            "rationale": "Neither offline-SCP nor online-recal-SCP cells found.",
        }
        write_markdown_summary(pd.DataFrame(), verdict,
                               tables_dir / "online_recal_summary.md")
        return 0

    # Attach replicas / SLO / p95 from metrics.json so the comparison can
    # report deltas even when those columns aren't already on the cell row.
    off_run_metrics = _collect_run_metrics(offline_inputs)
    on_run_metrics = _collect_run_metrics(online_inputs)
    off_cells = _merge_with_metrics(off_cells, off_run_metrics)
    on_cells = _merge_with_metrics(on_cells, on_run_metrics)

    off_cells.to_csv(data_dir / "offline_scp_per_cell.csv", index=False)
    on_cells.to_csv(data_dir / "online_recal_scp_per_cell.csv", index=False)

    comparison = build_comparison_table(off_cells, on_cells)
    comparison.to_csv(data_dir / "online_vs_offline_per_workload.csv", index=False)

    closure = overall_closure(comparison)
    verdict = narrative_verdict(closure)
    (data_dir / "narrative_verdict.json").write_text(
        pd.Series(verdict).to_json(indent=2)
    )

    write_coverage_figure(comparison, fig_dir / "fig_online_vs_offline_coverage.pdf")
    write_markdown_summary(comparison, verdict,
                           tables_dir / "online_recal_summary.md")

    print("\n── Open-Q7 verdict ──")
    print(f"  Overall closure : {closure*100:.1f}%" if np.isfinite(closure)
          else "  Overall closure : undefined (online data missing)")
    print(f"  Framing         : {verdict['framing']}")
    print(f"  Rationale       : {verdict['rationale']}")

    if not comparison.empty:
        print("\nPer-workload comparison:")
        pd.options.display.float_format = "{:.3f}".format
        print(comparison.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
