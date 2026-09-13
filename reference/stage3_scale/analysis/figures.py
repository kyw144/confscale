"""Figures: generate publication-quality figures for the P3 paper.

Figures owned by this module:
    1. System Architecture Diagram (programmatic SVG)
    2. Workload Patterns with Predictions + CI Bands (4-panel)
    3. SLO Violation Rate Bar Chart
    4. Resource Overhead Bar Chart
    6. Time-Series Excerpt — Burst Event (3-panel)
    7. Time-in-Tier Distribution

Owned by sibling modules (so they read the right data sources):
    5. λ Pareto frontier — `analysis/lambda_analysis.py` (λ-sweep cells)
    8. Coverage Calibration Plot — `analysis/calibration.py` (scale-log post-hoc)

All figures save at 300 DPI to PDF (vector-compatible) with consistent styling.
Uses seaborn colorblind palette for accessibility.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .loader import load_runs, load_timeseries
from .metrics import (
    aggregate_by_method_workload,
    compute_efficiency_score,
    compute_slo_violation_rate,
)

logger = logging.getLogger(__name__)

# ── Global Styling ──────────────────────────────────────────────────────────

METHOD_ORDER = [
    "hpa-reactive", "hpa-predictive", "hpa-predictive-safety",
    "keda", "base-inspired",
    "confscale-be", "confscale-scp", "confscale-qr",
]

WORKLOAD_ORDER = ["A", "B", "C", "D"]

METHOD_LABELS = {
    "hpa-reactive": "HPA-Reactive",
    "hpa-predictive": "HPA-Predictive",
    "hpa-predictive-safety": "HPA-Pred-Safety",
    "keda": "KEDA",
    "base-inspired": "BASE-Insp.",
    "confscale-be": "ConfScale-BE",
    "confscale-scp": "ConfScale-SCP",
    "confscale-qr": "ConfScale-QR",
}

WORKLOAD_LABELS = {"A": "Diurnal", "B": "Bursty", "C": "Batch-Ramp", "D": "Signaling"}

METHOD_COLORS = {
    "hpa-reactive": "#1f77b4",
    "hpa-predictive": "#ff7f0e",
    "hpa-predictive-safety": "#d62728",
    "keda": "#9467bd",
    "base-inspired": "#8c564b",
    "confscale-be": "#e377c2",
    "confscale-scp": "#2ca02c",
    "confscale-qr": "#17becf",
}

TIER_COLORS = {"tier1": "#27ae60", "tier2": "#f39c12", "tier3": "#e74c3c"}

WORKLOAD_COLORS = {"A": "#3498db", "B": "#e74c3c", "C": "#2ecc71", "D": "#9b59b6"}

FIGURE_DPI = 300


def _setup_style():
    """Configure matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    sns.set_palette("colorblind")


# ── Figure 1: System Architecture Diagram ───────────────────────────────────

def generate_figure_1(output_path: str) -> str:
    """Figure 1: System Architecture — MAPE-K loop with confidence-aware tiered policy.

    This is a programmatic diagram (not data-driven), showing the MAPE-K
    control loop with the confidence tier system.
    """
    _setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Color scheme
    c_blue = "#2c3e50"
    c_green = "#27ae60"
    c_orange = "#f39c12"
    c_red = "#e74c3c"
    c_gray = "#95a5a6"
    c_light = "#ecf0f1"

    def draw_box(x, y, w, h, label, color, text_color="white", fontsize=8):
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.15", facecolor=color, edgecolor="black",
            linewidth=1, alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", color=text_color,
                fontsize=fontsize, fontweight="bold")

    def draw_arrow(x1, y1, x2, y2, label="", color=c_gray):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                          connectionstyle="arc3,rad=0"),
        )
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.15, label, ha="center", va="bottom",
                   fontsize=6, color=color, fontstyle="italic")

    # MAPE-K Loop (Monitor → Analyze → Plan → Execute → Knowledge)
    draw_box(1, 5, 1.8, 0.9, "Monitor\n(Prometheus)", c_gray, fontsize=7)
    draw_box(1, 3.5, 1.8, 0.9, "Analyze\n(UQ Predict)", c_gray, fontsize=7)
    draw_box(1, 2, 1.8, 0.9, "Plan\n(Tier Policy)", c_gray, fontsize=7)
    draw_box(1, 0.5, 1.8, 0.9, "Execute\n(K8s Scale)", c_gray, fontsize=7)

    # Vertical arrows
    for y1, y2 in [(4.55, 3.95), (3.05, 2.45), (1.55, 0.95)]:
        draw_arrow(1, y1, 1, y2, color=c_gray)

    # Knowledge base on right
    draw_box(3.5, 5, 1.8, 3.5, "Knowledge\n\n• SLO Model\n• Capacity\n  Curves\n• Service\n  Profile", c_blue, fontsize=7)

    # Tier system on far right
    draw_box(6, 5, 1.5, 0.9, "Tier 1\nLow Uncertainty", c_green, fontsize=7)
    draw_box(6, 3.5, 1.5, 0.9, "Tier 2\nModerate Uncertainty", c_orange, fontsize=7)
    draw_box(6, 2, 1.5, 0.9, "Tier 3\nHigh Uncertainty", c_red, fontsize=7)

    # Tier transition arrow
    ax.annotate("", xy=(5.25, 5), xytext=(5.25, 2),
                arrowprops=dict(arrowstyle="<->", color=c_gray, lw=1))

    # UQ Methods box
    draw_box(8.5, 5, 1.5, 0.7, "BE", "#e377c2", fontsize=7)
    draw_box(8.5, 3.8, 1.5, 0.7, "SCP", "#2ca02c", fontsize=7)
    draw_box(8.5, 2.6, 1.5, 0.7, "QR", "#17becf", fontsize=7)

    # Feedback loop
    ax.annotate(
        "", xy=(1.9, 5), xytext=(0.1, 0.5),
        arrowprops=dict(arrowstyle="->", color=c_gray, lw=1,
                      connectionstyle="arc3,rad=-0.4"),
    )
    ax.text(0.65, 2.75, "Feedback", ha="center", fontsize=6, color=c_gray,
            fontstyle="italic", rotation=50)

    # Title
    ax.set_title("MAPE-K Architecture with Confidence-Aware Tiered Scaling",
                fontsize=12, fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Figure 1 saved: %s", output_path)
    return output_path


# ── Figure 2: Workload Patterns with Predictions ────────────────────────────

def generate_figure_2(
    input_dir: Path,
    output_path: str,
    method: str = "confscale-scp",
) -> str:
    """Figure 2: 4-panel time series showing workload patterns with predictions.

    Shows actual RPS, GRU forecast, and confidence intervals for each workload.
    """
    _setup_style()
    ts_df = load_timeseries(input_dir)

    if len(ts_df) == 0:
        logger.warning("No timeseries data for Figure 2")
        return _generate_placeholder_figure(output_path, "Figure 2: Workload Patterns\n(No data available)")

    workloads_present = [w for w in WORKLOAD_ORDER if w in ts_df["workload"].unique()]

    if not workloads_present:
        logger.warning("No workload data for Figure 2")
        return _generate_placeholder_figure(output_path, "Figure 2: Workload Patterns\n(No data available)")

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    axes = axes.flatten()

    for i, wl in enumerate(workloads_present[:4]):
        ax = axes[i]

        # Get one replicate for this workload
        wl_data = ts_df[(ts_df["workload"] == wl) & (ts_df["method"] == method)]
        if len(wl_data) == 0:
            wl_data = ts_df[ts_df["workload"] == wl]

        if len(wl_data) == 0:
            ax.text(0.5, 0.5, f"No data for {wl}", ha="center", va="center",
                   transform=ax.transAxes)
            ax.set_title(WORKLOAD_LABELS.get(wl, wl))
            continue

        # Take first run
        run_id = wl_data["run_id"].iloc[0]
        run_data = wl_data[wl_data["run_id"] == run_id].sort_values("timestamp")

        # Normalize timestamps to relative seconds
        if len(run_data) > 0:
            t0 = run_data["timestamp"].iloc[0]
            t = run_data["timestamp"] - t0

            # Plot actual RPS
            ax.plot(t, run_data["rps"], color=WORKLOAD_COLORS.get(wl, "#3498db"),
                   linewidth=0.8, alpha=0.8, label="Actual RPS")

            # Plot replicas on twin axis
            ax2 = ax.twinx()
            ax2.plot(t, run_data["replicas"], color="orange", linewidth=0.6,
                    alpha=0.5, linestyle="--", label="Replicas")
            ax2.set_ylabel("Replicas", fontsize=7, color="orange")
            ax2.tick_params(axis="y", labelsize=6, colors="orange")

            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("RPS", fontsize=8)

        ax.set_title(WORKLOAD_LABELS.get(wl, wl), fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Workload Patterns — Actual RPS and Replica Response",
                fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Figure 2 saved: %s", output_path)
    return output_path


# ── Figure 3: SLO Violation Rate Bar Chart ─────────────────────────────────

def generate_figure_3(df: pd.DataFrame, output_path: str) -> str:
    """Figure 3: Grouped bar chart of SLO violation rate per method × workload."""
    _setup_style()

    df = compute_slo_violation_rate(df)

    if len(df) == 0:
        return _generate_placeholder_figure(output_path, "Figure 3: SLO Violation Rate\n(No data available)")

    methods_present = [m for m in METHOD_ORDER if m in df["method"].unique()]
    workloads_present = [w for w in WORKLOAD_ORDER if w in df["workload"].unique()]

    if not methods_present or not workloads_present:
        return _generate_placeholder_figure(output_path, "Figure 3: SLO Violation Rate\n(Insufficient data)")

    # Compute means and standard errors
    agg = df.groupby(["method", "workload"]).agg(
        mean=("slo_violation_rate", "mean"),
        std=("slo_violation_rate", "std"),
        n=("slo_violation_rate", "count"),
    ).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["n"])

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(methods_present))
    n_workloads = len(workloads_present)
    width = 0.8 / n_workloads

    for i, wl in enumerate(workloads_present):
        wl_data = agg[agg["workload"] == wl]
        means = []
        errs = []
        for m in methods_present:
            row = wl_data[wl_data["method"] == m]
            if len(row) > 0:
                means.append(row["mean"].values[0])
                errs.append(row["sem"].values[0])
            else:
                means.append(0)
                errs.append(0)

        offset = (i - (n_workloads - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width,
            yerr=errs, capsize=2,
            label=WORKLOAD_LABELS.get(wl, wl),
            color=list(WORKLOAD_COLORS.values())[i],
            alpha=0.85,
        )

    # HPA-reactive reference line
    hpa_means = agg[agg["method"] == "hpa-reactive"].groupby("workload")["mean"].mean()
    if len(hpa_means) > 0:
        ref_val = hpa_means.mean()
        ax.axhline(y=ref_val, color="red", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(len(methods_present) - 0.5, ref_val, "HPA-Reactive mean",
               fontsize=7, color="red", va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods_present],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("SLO Violation Rate", fontsize=9)
    ax.set_title("SLO Violation Rate per Method × Workload", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Figure 3 saved: %s", output_path)
    return output_path


# ── Figure 4: Resource Overhead Bar Chart ──────────────────────────────────

def generate_figure_4(df: pd.DataFrame, output_path: str) -> str:
    """Figure 4: Resource Overhead per method × workload."""
    _setup_style()

    if len(df) == 0:
        return _generate_placeholder_figure(output_path, "Figure 4: Resource Overhead\n(No data available)")

    methods_present = [m for m in METHOD_ORDER if m in df["method"].unique()]
    workloads_present = [w for w in WORKLOAD_ORDER if w in df["workload"].unique()]

    if not methods_present:
        return _generate_placeholder_figure(output_path, "Figure 4: Resource Overhead\n(Insufficient data)")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, m in enumerate(methods_present):
        m_data = df[df["method"] == m]
        if "overhead_replica_seconds" not in m_data.columns:
            continue

        for j, wl in enumerate(workloads_present):
            wl_data = m_data[m_data["workload"] == wl]
            if len(wl_data) == 0:
                continue

            overheads = wl_data["overhead_replica_seconds"].dropna()
            if len(overheads) == 0:
                continue

            mean_val = overheads.mean()
            ax.bar(i, mean_val, width=0.8 / len(workloads_present),
                  bottom=sum(
                      m_data[m_data["workload"] == prev_wl]["overhead_replica_seconds"].mean()
                      for prev_wl in workloads_present[:j]
                      if prev_wl in m_data["workload"].values
                  ),
                  label=WORKLOAD_LABELS.get(wl, wl) if i == 0 else "",
                  color=list(WORKLOAD_COLORS.values())[j % len(WORKLOAD_COLORS)],
                  alpha=0.85)

    # Use a cleaner grouped approach
    plt.close(fig)

    # Redo with proper grouped bars
    agg = df.groupby(["method", "workload"]).agg(
        overhead_mean=("overhead_replica_seconds", "mean"),
        overhead_std=("overhead_replica_seconds", "std"),
        n=("overhead_replica_seconds", "count"),
    ).reset_index()
    agg["sem"] = agg["overhead_std"] / np.sqrt(agg["n"])

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(methods_present))
    n_workloads = len(workloads_present)
    width = 0.8 / n_workloads

    for i, wl in enumerate(workloads_present):
        wl_data = agg[agg["workload"] == wl]
        means = []
        errs = []
        for m in methods_present:
            row = wl_data[wl_data["method"] == m]
            if len(row) > 0:
                means.append(row["overhead_mean"].values[0])
                errs.append(row["sem"].values[0])
            else:
                means.append(0)
                errs.append(0)

        offset = (i - (n_workloads - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=errs, capsize=2,
               label=WORKLOAD_LABELS.get(wl, wl),
               color=list(WORKLOAD_COLORS.values())[i],
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods_present],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Excess Replica-Seconds", fontsize=9)
    ax.set_title("Resource Overhead per Method × Workload", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Figure 4 saved: %s", output_path)
    return output_path


# ── Figure 5 (λ Pareto) lives in analysis/lambda_analysis.py ────────────────
# It depends on the dedicated λ-sweep output directories
# (`outputs/p3_lambda_sweep_*`), which `report.py` does not load.


# ── Figure 6: Time-Series Excerpt (Burst Event) ─────────────────────────────

def generate_figure_6(
    input_dir: Path,
    output_path: str,
    method: str = "confscale-scp",
    workload: str = "B",
) -> str:
    """Figure 6: 3-panel time series of a burst event.

    Shows RPS + predictions (top), replica response (middle), p95 latency (bottom).
    """
    _setup_style()

    ts_df = load_timeseries(input_dir)

    if len(ts_df) == 0:
        return _generate_placeholder_figure(output_path, "Figure 6: Burst Event Excerpt\n(No data available)")

    # Get bursty workload for the specified method
    wl_data = ts_df[(ts_df["workload"] == workload) & (ts_df["method"] == method)]
    if len(wl_data) == 0:
        wl_data = ts_df[ts_df["workload"] == workload]

    if len(wl_data) == 0:
        return _generate_placeholder_figure(output_path, "Figure 6: Burst Event Excerpt\n(No data available)")

    # Use first run, take a 5-minute window from the middle
    run_id = wl_data["run_id"].iloc[0]
    run_data = wl_data[wl_data["run_id"] == run_id].sort_values("timestamp")

    if len(run_data) < 20:
        return _generate_placeholder_figure(output_path,
                                           "Figure 6: Burst Event Excerpt\n(Insufficient data)")

    # Take roughly 20 data points (5 min at 15s intervals)
    n_show = min(20, len(run_data))
    start_idx = max(0, len(run_data) // 2 - n_show // 2)
    excerpt = run_data.iloc[start_idx:start_idx + n_show]

    t = np.arange(n_show) * 15  # 15s intervals

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    # Panel 1: RPS
    ax1.plot(t, excerpt["rps"].values, "b-", linewidth=1.2, label="Actual RPS")
    ax1.fill_between(t, 0, excerpt["rps"].values, alpha=0.15, color="blue")
    ax1.set_ylabel("RPS", fontsize=8)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Load (Requests/Second)", fontsize=9)

    # Panel 2: Replicas
    ax2.step(t, excerpt["replicas"].values, "orange", linewidth=1.5,
            where="post", label="Replicas")
    ax2.fill_between(t, 0, excerpt["replicas"].values, alpha=0.15, color="orange")
    ax2.set_ylabel("Replicas", fontsize=8)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Scaling Response (Replica Count)", fontsize=9)

    # Panel 3: Latency
    ax3.plot(t, excerpt["p95_ms"].values, "r-", linewidth=1.2, label="p95 Latency")
    ax3.axhline(y=200, color="gray", linestyle="--", linewidth=0.8,
               label="SLO (200ms)")
    ax3.fill_between(t, 0, excerpt["p95_ms"].values, alpha=0.15, color="red")
    ax3.set_ylabel("Latency (ms)", fontsize=8)
    ax3.set_xlabel("Time (seconds)", fontsize=8)
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("p95 End-to-End Latency", fontsize=9)

    fig.suptitle(f"Burst Event Excerpt — {METHOD_LABELS.get(method, method)} on "
                f"{WORKLOAD_LABELS.get(workload, workload)} Workload",
                fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Figure 6 saved: %s", output_path)
    return output_path


# ── Figure 7: Time-in-Tier Distribution ────────────────────────────────────

def generate_figure_7(
    df: pd.DataFrame,
    output_path: str,
) -> str:
    """Figure 7: Stacked bar chart of time spent in each tier per workload.

    Falls back to workload-level analysis if tier data is not directly available.
    """
    _setup_style()

    workloads_present = [w for w in WORKLOAD_ORDER if w in df["workload"].unique()]

    if not workloads_present:
        return _generate_placeholder_figure(output_path,
                                           "Figure 7: Time-in-Tier Distribution\n(No data available)")

    fig, ax = plt.subplots(figsize=(6, 4))

    # Check if tier data exists
    if "tier" in df.columns:
        for wl in workloads_present:
            wl_data = df[df["workload"] == wl]
            tier_counts = wl_data["tier"].value_counts(normalize=True).sort_index()

            bottom = 0
            for tier, pct in tier_counts.items():
                color_idx = min(int(tier) - 1, 2)
                color = list(TIER_COLORS.values())[color_idx]
                ax.barh(WORKLOAD_LABELS.get(wl, wl), pct, left=bottom,
                       color=color, alpha=0.85,
                       label=f"Tier {tier}" if wl == workloads_present[0] else "")
                bottom += pct
    else:
        # Fall back: classify workloads as tier proxies
        tier_map = {
            "A": ("Tier 1 (stable)", TIER_COLORS["tier1"]),
            "B": ("Tier 2/3 (bursty)", TIER_COLORS["tier2"]),
            "C": ("Tier 2/3 (transitional)", TIER_COLORS["tier2"]),
            "D": ("Tier 2/3 (intermittent)", TIER_COLORS["tier3"]),
        }

        for wl in workloads_present:
            label, color = tier_map.get(wl, (wl, "gray"))
            # Use relative SLO violation rate as a proxy for "tier time"
            wl_mean = df[df["workload"] == wl]["slo_violation_rate"].mean()
            ax.barh(WORKLOAD_LABELS.get(wl, wl), 1.0, color=color, alpha=0.6,
                   edgecolor="black", linewidth=0.5)

        # Create legend
        legend_items = [
            mpatches.Patch(color=TIER_COLORS["tier1"], label="Tier 1 (Stable/Predictable)"),
            mpatches.Patch(color=TIER_COLORS["tier2"], label="Tier 2 (Moderate Uncertainty)"),
            mpatches.Patch(color=TIER_COLORS["tier3"], label="Tier 3 (High Uncertainty)"),
        ]
        ax.legend(handles=legend_items, fontsize=7)

    ax.set_xlabel("Proportion of Time", fontsize=9)
    ax.set_title("Time-in-Tier Distribution by Workload", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Figure 7 saved: %s", output_path)
    return output_path


# ── Figure 8 (coverage calibration) lives in analysis/calibration.py ────────
# It joins controller_scale_log.json against timeseries.csv to compute
# empirical coverage; running it here from the runs DataFrame alone produced
# all-NaN output. Run `python -m paper3_experiments.analysis.calibration
# --input <merged_dir> --output <results_dir>` to regenerate.


# ── Helpers ─────────────────────────────────────────────────────────────────

def _generate_placeholder_figure(output_path: str, message: str) -> str:
    """Generate a placeholder figure when data is missing."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12,
           transform=ax.transAxes, color="gray")
    ax.axis("off")
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ── All Figures ─────────────────────────────────────────────────────────────

def generate_all_figures(
    df: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Generate all 8 figures.

    Args:
        df: Runs DataFrame (from loader)
        input_dir: Root directory with run outputs (for timeseries data)
        output_dir: Directory to save figures

    Returns:
        dict mapping figure names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Figure 1: Architecture diagram (no data needed)
    figures["fig_1_architecture"] = generate_figure_1(
        str(output_dir / "fig_1_architecture.pdf")
    )

    # Figure 2: Workload patterns (needs timeseries)
    figures["fig_2_workload_patterns"] = generate_figure_2(
        input_dir, str(output_dir / "fig_2_workload_patterns.pdf")
    )

    # Figure 3: SLO violation bars
    figures["fig_3_slo_bars"] = generate_figure_3(
        df, str(output_dir / "fig_3_slo_bars.pdf")
    )

    # Figure 4: Resource overhead bars
    figures["fig_4_overhead_bars"] = generate_figure_4(
        df, str(output_dir / "fig_4_overhead_bars.pdf")
    )

    # Figure 5 (λ Pareto): generated by analysis/lambda_analysis.py
    # Figure 6: Burst event excerpt
    figures["fig_6_burst_excerpt"] = generate_figure_6(
        input_dir, str(output_dir / "fig_6_burst_excerpt.pdf")
    )

    # Figure 7: Time-in-tier
    figures["fig_7_time_in_tier"] = generate_figure_7(
        df, str(output_dir / "fig_7_time_in_tier.pdf")
    )

    # Figure 8 (coverage calibration): generated by analysis/calibration.py
    return figures
