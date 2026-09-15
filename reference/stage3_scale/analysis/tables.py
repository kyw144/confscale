"""Export experiment tables as LaTeX and Markdown."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .loader import load_runs
from .metrics import (
    aggregate_by_method_workload,
    compute_efficiency_score,
    compute_lambda_sweep,
    compute_slo_violation_rate,
    compute_uq_comparison,
)
from .stats import compare_all_methods, PRIMARY_BASELINE

logger = logging.getLogger(__name__)


METHOD_ORDER = [
    "hpa-reactive", "hpa-predictive", "hpa-predictive-safety",
    "keda", "base-inspired",
    "confscale-be", "confscale-scp", "confscale-qr",
    "confscale-scp-online", "risk-quantile-scp",
]

WORKLOAD_ORDER = ["A", "B", "C", "D"]

METHOD_LABELS = {
    "hpa-reactive": "HPA-Reactive",
    "hpa-predictive": "HPA-Predictive",
    "hpa-predictive-safety": "HPA-Pred-Safety",
    "keda": "KEDA",
    "base-inspired": "BASE-Inspired",
    "confscale-be": "ConfScale-BE",
    "confscale-scp": "ConfScale-SCP",
    "confscale-qr": "ConfScale-QR",
    "confscale-scp-online": "ConfScale-SCP-Online",
    "risk-quantile-scp": "Risk-Quantile-SCP",
}

WORKLOAD_LABELS = {"A": "Diurnal", "B": "Bursty", "C": "Batch-Ramp", "D": "Signaling"}


def _get_significance_annotation(
    method: str,
    workload: str,
    comparisons: list[dict],
) -> str:
    for c in comparisons:
        if c.get("method_a") == method and c.get("workload") == workload:
            if not c.get("significant_corrected"):
                return ""
            d = abs(c.get("cohens_d", 0))
            p = c.get("p_value_corrected", 1)

            if p < 0.001 and d > 0.8:
                return "§"
            elif p < 0.01 and d > 0.8:
                return "‡"
            elif p < 0.05 and d > 0.5:
                return "†"
            elif p < 0.05:
                return "*"
            return ""
    return ""


def _escape_latex(s: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }
    for char, repl in replacements.items():
        s = s.replace(char, repl)
    return s


def _df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_format: str = ".3f",
    notes: str = "",
) -> str:
    n_cols = len(df.columns) + 1  # +1 for row labels
    col_format = "l" + "c" * len(df.columns)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{col_format}}}",
        r"    \toprule",
    ]

    header = " & ".join(["Method"] + [f"\\textbf{{{_escape_latex(str(c))}}}" for c in df.columns])
    lines.append(f"    {header} \\\\")
    lines.append(r"    \midrule")

    for idx, row in df.iterrows():
        vals = [str(idx)]
        for col in df.columns:
            val = row[col]
            if isinstance(val, float) or isinstance(val, (np.floating,)):
                if np.isnan(float(val)):
                    vals.append("—")
                else:
                    vals.append(f"{float(val):{float_format}}")
            elif isinstance(val, (int, np.integer)):
                vals.append(str(val))
            else:
                vals.append("—")
        lines.append(f"    {' & '.join(vals)} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")

    if notes:
        lines.append(f"  {notes}")

    lines.append(r"\end{table}")
    return "\n".join(lines)


def _df_to_markdown(df: pd.DataFrame, caption: str, float_format: str = ".3f") -> str:
    lines = [f"### {caption}", ""]

    header = "| Method | " + " | ".join(str(c) for c in df.columns) + " |"
    lines.append(header)

    sep = "|" + "|".join("---" for _ in range(len(df.columns) + 1)) + "|"
    lines.append(sep)

    for idx, row in df.iterrows():
        vals = [str(idx)]
        for col in df.columns:
            val = row[col]
            if isinstance(val, float) or isinstance(val, (np.floating,)):
                if np.isnan(float(val)):
                    vals.append("—")
                else:
                    vals.append(f"{float(val):{float_format}}")
            elif isinstance(val, (int, np.integer)):
                vals.append(str(val))
            elif isinstance(val, str) and val:
                vals.append(val)
            else:
                vals.append("—")
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_table_1(
    df: pd.DataFrame,
    output_dir: Path,
    comparisons: Optional[list[dict]] = None,
) -> dict[str, Path]:
    """Table 1: SLO Violation Rate per Method × Workload."""
    df = compute_slo_violation_rate(df)

    if comparisons is None:
        comp = compare_all_methods(df)
        comparisons = comp.get("all_comparisons", [])

    # Pivot: rows=method, columns=workload, values=mean(slo_violation_rate)
    pivot = df.pivot_table(
        values="slo_violation_rate",
        index="method",
        columns="workload",
        aggfunc="mean",
    )

    methods_present = [m for m in METHOD_ORDER if m in pivot.index]
    workloads_present = [w for w in WORKLOAD_ORDER if w in pivot.columns]

    if not methods_present or not workloads_present:
        logger.warning("Insufficient data for Table 1")
        return {}

    pivot = pivot.loc[methods_present, workloads_present]

    pivot["Mean"] = pivot.mean(axis=1)

    pivot.columns = [WORKLOAD_LABELS.get(c, c) for c in pivot.columns]

    # Annotate with significance (cast to object so float columns accept strings)
    annotated = pivot.astype(object).copy()
    for method in methods_present:
        for wl in workloads_present:
            ann = _get_significance_annotation(method, wl, comparisons)
            if ann:
                val = pivot.loc[method, WORKLOAD_LABELS.get(wl, wl)]
                annotated.loc[method, WORKLOAD_LABELS.get(wl, wl)] = f"{val:.3f}{ann}"

    annotated.index = [METHOD_LABELS.get(m, m) for m in annotated.index]

    if len(pivot) > 0:
        latex_df = pivot.copy()
        latex_df.index = [METHOD_LABELS.get(m, m) for m in latex_df.index]

        latex = _df_to_latex(
            latex_df,
            caption="SLO Violation Rate per Method × Workload (mean of 3 replicates)",
            label="tab:slo_violation_rate",
            notes=r"  \caption*{\small Bold = best per workload. "
                  r"† $p_{corr} < 0.05$, § $p_{corr} < 0.001$. "
                  r"Corrected with Bonferroni-Holm.}",
        )
    else:
        latex = "% Table 1: No data available\n"

    md = _df_to_markdown(annotated, "Table 1: SLO Violation Rate per Method × Workload")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latex_path = output_dir / "table_1_slo_violations.tex"
    md_path = output_dir / "table_1_slo_violations.md"

    latex_path.write_text(latex)
    md_path.write_text(md)

    logger.info("Table 1 written: %s, %s", latex_path, md_path)
    return {"latex": latex_path, "markdown": md_path}


def generate_table_2(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Table 2: Resource Overhead (excess replica-seconds) per Method × Workload."""
    pivot = df.pivot_table(
        values="overhead_replica_seconds",
        index="method",
        columns="workload",
        aggfunc="mean",
    )

    methods_present = [m for m in METHOD_ORDER if m in pivot.index]
    workloads_present = [w for w in WORKLOAD_ORDER if w in pivot.columns]

    if not methods_present or not workloads_present:
        logger.warning("Insufficient data for Table 2")
        return {}

    pivot = pivot.loc[methods_present, workloads_present]
    pivot["Mean"] = pivot.mean(axis=1)
    pivot.columns = [WORKLOAD_LABELS.get(c, c) for c in pivot.columns]

    replicas_pivot = df.pivot_table(
        values="mean_replicas",
        index="method",
        columns="workload",
        aggfunc="mean",
    )
    if all(m in replicas_pivot.index for m in methods_present):
        replicas_pivot = replicas_pivot.loc[methods_present, workloads_present]

    latex_df = pivot.copy()
    latex_df.index = [METHOD_LABELS.get(m, m) for m in latex_df.index]

    latex_df = latex_df.round(0).astype(int)

    latex = _df_to_latex(
        latex_df,
        caption="Resource Overhead (excess replica-seconds) per Method × Workload",
        label="tab:resource_overhead",
        float_format=".0f",
        notes=r"  \caption*{\small Overhead = excess replica-seconds beyond ideal minimal count. "
              r"Lower is better.}",
    )

    annotated = latex_df.copy()
    md = _df_to_markdown(annotated, "Table 2: Resource Overhead (excess replica-seconds)", float_format="%.0f")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latex_path = output_dir / "table_2_resource_overhead.tex"
    md_path = output_dir / "table_2_resource_overhead.md"

    latex_path.write_text(latex)
    md_path.write_text(md)

    logger.info("Table 2 written: %s, %s", latex_path, md_path)
    return {"latex": latex_path, "markdown": md_path}


def generate_table_3(
    df: pd.DataFrame,
    output_dir: Path,
    input_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Table 3: UQ Method Comparison — coverage, CI width, efficiency, inference."""
    uq_df = compute_uq_comparison(df, input_dir=input_dir)

    if len(uq_df) == 0:
        logger.warning("No UQ methods for Table 3")
        return {}

    cols = {
        "coverage": "Coverage",
        "median_ci_width": "Median CI Width (RPS)",
        "mean_ci_width": "Mean CI Width (RPS)",
        "efficiency_score": "Efficiency Score",
        "inference_ms": "Inference (ms)",
    }
    available = {k: v for k, v in cols.items() if k in uq_df.columns}

    display_df = uq_df[list(available.keys())].copy()
    display_df.index = [METHOD_LABELS.get(m, m) for m in display_df.index]
    display_df.columns = [available[c] for c in display_df.columns]

    if "Coverage" in display_df.columns:
        display_df["Coverage"] = display_df["Coverage"].round(3)
    if "Efficiency Score" in display_df.columns:
        display_df["Efficiency Score"] = display_df["Efficiency Score"].round(3)
    if "Median CI Width (RPS)" in display_df.columns:
        display_df["Median CI Width (RPS)"] = display_df["Median CI Width (RPS)"].round(2)
    if "Mean CI Width (RPS)" in display_df.columns:
        display_df["Mean CI Width (RPS)"] = display_df["Mean CI Width (RPS)"].round(2)
    if "Inference (ms)" in display_df.columns:
        display_df["Inference (ms)"] = display_df["Inference (ms)"].round(1)

    latex_df = display_df.copy()
    latex = _df_to_latex(
        latex_df,
        caption=r"UQ Method Comparison — Coverage (target 90\%), Interval Width, Efficiency, "
                r"and Controller Decision Latency",
        label="tab:uq_comparison",
        float_format=".3f",
        notes=r"  \caption*{\small Coverage is post-hoc empirical, target 0.90. "
              r"Efficiency Score = (1 - SLO-Violation-Rate) / (Overhead-Hours + 1). "
              r"Inference latency is the mean controller decision time (ms) over all "
              r"scale events.}",
    )

    md = _df_to_markdown(display_df, "Table 3: UQ Method Comparison")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latex_path = output_dir / "table_3_uq_comparison.tex"
    md_path = output_dir / "table_3_uq_comparison.md"

    latex_path.write_text(latex)
    md_path.write_text(md)

    logger.info("Table 3 written: %s, %s", latex_path, md_path)
    return {"latex": latex_path, "markdown": md_path}


def generate_table_4(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Table 4: Lambda Sensitivity Sweep — SLO rate vs. overhead across λ values."""
    sweep = compute_lambda_sweep(df)

    if sweep is None or len(sweep) == 0:
        logger.warning("No lambda sweep data for Table 4")
        return {}

    display_df = sweep.set_index("lambda").copy()
    display_df = display_df.rename(columns={
        "slo_violation_rate": "SLO Violation Rate",
        "overhead_replica_seconds": "Resource Overhead",
        "mean_replicas": "Mean Replicas",
    })

    display_df["SLO Violation Rate"] = display_df["SLO Violation Rate"].round(3)
    display_df["Resource Overhead"] = display_df["Resource Overhead"].round(0).astype(int)
    display_df["Mean Replicas"] = display_df["Mean Replicas"].round(1)

    latex_df = display_df[["SLO Violation Rate", "Resource Overhead", "Mean Replicas"]].copy()
    latex = _df_to_latex(
        latex_df,
        caption=r"$\lambda$ Sensitivity Sweep — SLO Violation Rate vs. Resource Overhead",
        label="tab:lambda_sweep",
        notes=r"  \caption*{\small Higher $\lambda$ = more conservative scaling. "
              r"Ideal operating point balances low violations with low overhead.}",
    )

    md = _df_to_markdown(display_df, "Table 4: λ Sensitivity Sweep")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latex_path = output_dir / "table_4_lambda_sweep.tex"
    md_path = output_dir / "table_4_lambda_sweep.md"

    latex_path.write_text(latex)
    md_path.write_text(md)

    logger.info("Table 4 written: %s, %s", latex_path, md_path)
    return {"latex": latex_path, "markdown": md_path}


def generate_all_tables(
    df: pd.DataFrame,
    output_dir: Path,
    input_dir: Optional[Path] = None,
) -> dict[str, dict[str, Path]]:
    """Generate all 4 tables."""
    # Pre-compute comparisons for Table 1 annotations
    comparisons = compare_all_methods(df)
    all_comps = comparisons.get("all_comparisons", [])

    return {
        "table_1_slo_violations": generate_table_1(df, output_dir, all_comps),
        "table_2_resource_overhead": generate_table_2(df, output_dir),
        "table_3_uq_comparison": generate_table_3(df, output_dir, input_dir=input_dir),
        "table_4_lambda_sweep": generate_table_4(df, output_dir),
    }
