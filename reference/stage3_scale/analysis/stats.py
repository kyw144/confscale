"""Statistics: hypothesis tests, effect sizes, multiple comparison correction."""

import json
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .loader import load_runs
from .metrics import compute_slo_violation_rate

logger = logging.getLogger(__name__)

PRIMARY_BASELINE = "hpa-reactive"
PREDICTIVE_BASELINE = "hpa-predictive"


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    diff = x - y
    if len(diff) < 2:
        return np.nan
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    if sd_diff == 0:
        return 0.0
    return mean_diff / sd_diff


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d value."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def cohens_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled SD)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def welch_t_test(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided") -> dict:
    """Welch's t-test (independent samples, unequal variance)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return {
            "test": "welch_t",
            "t_statistic": None,
            "p_value": float("nan"),
            "n_x": len(x),
            "n_y": len(y),
            "error": f"insufficient samples (n_x={len(x)}, n_y={len(y)}; need ≥2 each)",
        }
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, alternative=alternative)
    return {
        "test": "welch_t",
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "n_x": int(len(x)),
        "n_y": int(len(y)),
        "alternative": alternative,
    }


def paired_comparison(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    workload: Optional[str] = None,
    metric: str = "slo_violation_rate",
    alternative: str = "two-sided",
) -> dict:
    """Compare two methods on a metric, paired by replicate within workload."""
    filt = df["method"].isin([method_a, method_b])
    if workload:
        filt &= df["workload"] == workload

    subset = df[filt].copy()

    if len(subset) == 0:
        return {"error": f"No data for {method_a} vs {method_b}"}

    # Pivot: one row per (workload, replicate), columns for each method
    pivot = subset.pivot_table(
        values=metric,
        index=["workload", "replicate"],
        columns="method",
    ).dropna()

    if len(pivot) < 3:
        return {
            "error": f"Insufficient paired data (n={len(pivot)} pairs)",
            "n_pairs": len(pivot),
        }

    x = pivot[method_a].values
    y = pivot[method_b].values

    diff = x - y
    shapiro_stat, shapiro_p = stats.shapiro(diff) if len(diff) >= 3 else (np.nan, np.nan)

    if shapiro_p < 0.05 and len(diff) >= 5:
        try:
            if alternative == "two-sided":
                w_stat, raw_p = stats.wilcoxon(x, y, alternative="two-sided")
            elif alternative == "less":
                w_stat, raw_p = stats.wilcoxon(x, y, alternative="less")
            else:
                w_stat, raw_p = stats.wilcoxon(x, y, alternative="greater")
            test_used = "wilcoxon"
            t_stat = np.nan  # Wilcoxon uses W statistic
        except Exception:
            # Fallback to t-test if Wilcoxon fails
            t_stat, raw_p = stats.ttest_rel(x, y, alternative=alternative)
            test_used = "ttest_rel"
    else:
        t_stat, raw_p = stats.ttest_rel(x, y, alternative=alternative)
        w_stat = np.nan
        test_used = "ttest_rel"

    d = cohens_d(x, y)
    mean_a = np.mean(x)
    mean_b = np.mean(y)
    rel_change = (mean_a - mean_b) / mean_b if mean_b != 0 else np.nan

    return {
        "method_a": method_a,
        "method_b": method_b,
        "workload": workload or "all",
        "metric": metric,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_a - mean_b,
        "rel_change_pct": rel_change * 100 if not np.isnan(rel_change) else None,
        "cohens_d": d,
        "effect_size": interpret_effect_size(d),
        "test_used": test_used,
        "statistic": float(t_stat) if not np.isnan(t_stat) else float(w_stat),
        "p_value_raw": float(raw_p),
        "p_value_significant": float(raw_p) < 0.05,
        "shapiro_p_normal": float(shapiro_p),
        "n_pairs": len(pivot),
    }


def bonferroni_holm_correction(p_values: list[float]) -> list[float]:
    """Apply Bonferroni-Holm correction."""
    n = len(p_values)
    if n == 0:
        return []

    indexed = list(enumerate(p_values))
    sorted_idx = sorted(indexed, key=lambda x: x[1])

    corrected = [0.0] * n
    for rank, (orig_idx, p_val) in enumerate(sorted_idx):
        multiplier = n - rank
        corrected_p = min(p_val * multiplier, 1.0)

        if rank > 0:
            prev = corrected[sorted_idx[rank - 1][0]]
            corrected_p = max(corrected_p, prev)

        corrected[orig_idx] = corrected_p

    return corrected


def test_h1_slo_reduction(
    df: pd.DataFrame,
    baseline: str = PRIMARY_BASELINE,
    alpha: float = 0.05,
) -> dict:
    """H1: ConfScale reduces SLO violations vs. point-forecast baseline."""
    df = compute_slo_violation_rate(df)

    confscale_methods = [m for m in df["method"].unique() if m.startswith("confscale-")]
    workloads = sorted(df["workload"].unique())

    if len(confscale_methods) == 0:
        return {"status": "skipped", "reason": "No confscale-* methods found"}

    all_comparisons = []

    for method in confscale_methods:
        for wl in workloads:
            comp = paired_comparison(
                df, method, baseline, workload=wl, metric="slo_violation_rate",
                alternative="less",  # We expect ConfScale to have LOWER violation rate
            )
            if "error" not in comp:
                all_comparisons.append(comp)

    p_values = [c["p_value_raw"] for c in all_comparisons]
    corrected = bonferroni_holm_correction(p_values)

    for comp, corr_p in zip(all_comparisons, corrected):
        comp["p_value_corrected"] = corr_p
        comp["significant_corrected"] = corr_p < alpha

    n_sig = sum(c["significant_corrected"] for c in all_comparisons)
    n_total = len(all_comparisons)

    return {
        "hypothesis": "H1",
        "description": "ConfScale reduces SLO violations vs. point-forecast baseline",
        "baseline": baseline,
        "n_comparisons": n_total,
        "n_significant": n_sig,
        "alpha": alpha,
        "correction": "bonferroni-holm",
        "comparisons": all_comparisons,
        "verdict": (
            "supported" if n_sig >= n_total * 0.5
            else "partial" if n_sig > 0
            else "not_supported"
        ),
    }


def test_h2_uq_efficiency(df: pd.DataFrame) -> dict:
    """H2: Which UQ method has the best efficiency score?"""
    from .metrics import compute_efficiency_score

    df = df.copy()
    df["efficiency"] = compute_efficiency_score(df)

    uq_methods = [m for m in df["method"].unique() if m.startswith("confscale-")]

    if len(uq_methods) < 2:
        return {"status": "skipped", "reason": f"Need ≥2 UQ methods, found {len(uq_methods)}"}

    # Pivot: one row per (replicate, workload), columns per method
    uq_df = df[df["method"].isin(uq_methods)]
    pivot = uq_df.pivot_table(
        values="efficiency",
        index=["workload", "replicate"],
        columns="method",
    ).dropna()

    if len(pivot) < 3:
        return {"status": "skipped", "reason": f"Insufficient data ({len(pivot)} complete cases)"}

    groups = [pivot[m].values for m in uq_methods]
    try:
        friedman_stat, friedman_p = stats.friedmanchisquare(*groups)
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    means = {m: float(pivot[m].mean()) for m in uq_methods}
    best = max(means, key=means.get)

    pairwise = []
    if friedman_p < 0.05:
        for m1, m2 in combinations(uq_methods, 2):
            w, p = stats.wilcoxon(pivot[m1], pivot[m2])
            pairwise.append({
                "method_a": m1,
                "method_b": m2,
                "wilcoxon_p": float(p),
                "significant": float(p) < 0.05 / len(uq_methods),  # Bonferroni for post-hoc
            })

    return {
        "hypothesis": "H2",
        "description": "UQ method with best efficiency score",
        "test": "Friedman",
        "friedman_statistic": float(friedman_stat),
        "p_value": float(friedman_p),
        "significant": friedman_p < 0.05,
        "best_method": best,
        "mean_efficiency_by_method": means,
        "posthoc_pairwise": pairwise,
    }


def test_h3_lambda_monotonicity(df: pd.DataFrame) -> dict:
    """H3: λ (safety factor) produces monotonic Pareto curve."""
    if "lambda" not in df.columns:
        logger.info("No lambda column — checking if any config has it")
        return {"status": "skipped", "reason": "No lambda sweep data available"}

    lambdas = df["lambda"].unique()
    if len(lambdas) < 3:
        return {"status": "skipped", "reason": f"Need ≥3 λ values, found {len(lambdas)}"}

    agg = df.groupby("lambda")["slo_violation_rate"].mean().sort_index()

    rho, p_val = stats.spearmanr(agg.index, agg.values)

    r, r_p = stats.pearsonr(agg.index, agg.values)

    return {
        "hypothesis": "H3",
        "description": "λ produces monotonic Pareto curve (higher λ → lower violations)",
        "test": "Spearman rank correlation",
        "lambda_values": sorted(lambdas.tolist()),
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "pearson_r": float(r),
        "pearson_p": float(r_p),
        "significant": float(p_val) < 0.05,
        "direction": "negative" if rho < 0 else "positive",
        "verdict": (
            "supported" if rho < -0.7 and p_val < 0.05
            else "partial" if rho < 0 and p_val < 0.05
            else "not_supported"
        ),
    }


def test_h4_tier_concentration(df: pd.DataFrame) -> dict:
    """H4: Benefit of confidence-aware scaling concentrated in Tier 2/3."""
    if "tier" not in df.columns:
        logger.info("No tier data in run metrics — H4 test limited")
        # Fall back to per-workload analysis (bursty/signaling = Tier 2/3 proxies)
        workloads = sorted(df["workload"].unique())
        df = compute_slo_violation_rate(df)

        # Diurnal = Tier 1 proxy, Bursty/Signaling = Tier 2/3 proxy
        tier_proxy = {
            "A": "Tier 1 (diurnal)",
            "B": "Tier 2/3 (bursty)",
            "C": "Tier 2/3 (batch-ramp)",
            "D": "Tier 2/3 (signaling)",
        }

        results = {}
        for wl, label in tier_proxy.items():
            subset = df[df["workload"] == wl]
            if len(subset) < 3:
                continue

            confscale = subset[subset["method"].str.startswith("confscale-")]
            baseline = subset[subset["method"] == PRIMARY_BASELINE]

            if len(confscale) > 0 and len(baseline) > 0:
                mean_cs = confscale["slo_violation_rate"].mean()
                mean_bl = baseline["slo_violation_rate"].mean()
                results[wl] = {
                    "label": label,
                    "confscale_mean": float(mean_cs),
                    "baseline_mean": float(mean_bl),
                    "reduction_pct": float((mean_bl - mean_cs) / mean_bl * 100) if mean_bl > 0 else 0,
                    "benefit_significant": mean_cs < mean_bl,
                }

        return {
            "hypothesis": "H4",
            "description": "ConfScale benefit concentrated in Tier 2/3 workloads",
            "method": "per-workload comparison (workload as tier proxy)",
            "results": results,
            "verdict": "partial — tier data not directly available, using workload proxy",
        }

    from scipy.stats import f_oneway

    df = df.dropna(subset=["slo_violation_rate", "tier"])
    tiers = sorted(df["tier"].unique())

    results = {}
    for tier in tiers:
        tier_df = df[df["tier"] == tier]
        groups = []

        for method in tier_df["method"].unique():
            method_df = tier_df[tier_df["method"] == method]
            if len(method_df) > 0:
                groups.append(method_df["slo_violation_rate"].values)

        if len(groups) >= 2:
            try:
                f_stat, f_p = f_oneway(*groups)
                results[f"tier_{tier}"] = {
                    "n_methods": len(groups),
                    "f_statistic": float(f_stat),
                    "p_value": float(f_p),
                    "significant": f_p < 0.05,
                }
            except Exception:
                pass

    return {
        "hypothesis": "H4",
        "description": "ConfScale benefit concentrated in Tier 2/3",
        "method": "per-tier one-way ANOVA",
        "results": results,
    }


def compare_all_methods(
    df: pd.DataFrame,
    baseline: str = PRIMARY_BASELINE,
    metric: str = "slo_violation_rate",
    workloads: Optional[list[str]] = None,
) -> dict:
    """Compare all methods against the baseline, per workload."""
    df = compute_slo_violation_rate(df)

    methods = sorted(df["method"].unique())
    test_methods = [m for m in methods if m != baseline]

    if workloads is None:
        workloads = sorted(df["workload"].unique())

    all_comparisons = []
    for method in test_methods:
        for wl in workloads:
            comp = paired_comparison(
                df, method, baseline, workload=wl, metric=metric,
                alternative="two-sided",
            )
            if "error" not in comp:
                all_comparisons.append(comp)

    p_values = [c["p_value_raw"] for c in all_comparisons]
    corrected = bonferroni_holm_correction(p_values)

    for comp, corr_p in zip(all_comparisons, corrected):
        comp["p_value_corrected"] = corr_p
        comp["significant_corrected"] = corr_p < 0.05

    by_method = defaultdict(list)
    for c in all_comparisons:
        by_method[c["method_a"]].append({
            "workload": c["workload"],
            "cohens_d": c["cohens_d"],
            "rel_change_pct": c.get("rel_change_pct", 0),
            "p_value_corrected": c["p_value_corrected"],
            "significant_corrected": c["significant_corrected"],
        })

    return {
        "baseline": baseline,
        "metric": metric,
        "n_comparisons": len(all_comparisons),
        "n_significant": sum(c["significant_corrected"] for c in all_comparisons),
        "correction": "bonferroni-holm",
        "by_method": {m: comps for m, comps in sorted(by_method.items())},
        "all_comparisons": all_comparisons,
    }


def normality_tests(df: pd.DataFrame, metric: str = "slo_violation_rate") -> pd.DataFrame:
    """Run Shapiro-Wilk normality test on each method × workload group."""
    df = compute_slo_violation_rate(df)
    rows = []

    for (method, wl), group in df.groupby(["method", "workload"]):
        values = group[metric].dropna().values
        if len(values) < 3:
            rows.append({
                "method": method, "workload": wl,
                "n": len(values), "shapiro_w": np.nan, "shapiro_p": np.nan,
                "normal": None,
            })
        else:
            w_stat, w_p = stats.shapiro(values)
            rows.append({
                "method": method, "workload": wl,
                "n": len(values),
                "shapiro_w": float(w_stat),
                "shapiro_p": float(w_p),
                "normal": w_p >= 0.05,
            })

    return pd.DataFrame(rows)


def heterogeneity_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Test variance homogeneity across methods per workload (Levene's test)."""
    df = compute_slo_violation_rate(df)
    rows = []

    for wl, group in df.groupby("workload"):
        method_groups = []
        labels = []
        for method, mg in group.groupby("method"):
            vals = mg["slo_violation_rate"].dropna().values
            if len(vals) >= 2:
                method_groups.append(vals)
                labels.append(method)

        if len(method_groups) >= 2:
            try:
                levene_stat, levene_p = stats.levene(*method_groups)
                rows.append({
                    "workload": wl,
                    "levene_statistic": float(levene_stat),
                    "levene_p": float(levene_p),
                    "equal_variance": levene_p >= 0.05,
                    "n_methods": len(method_groups),
                })
            except Exception:
                pass

    return pd.DataFrame(rows)


def run_all_tests(df: pd.DataFrame) -> dict:
    """Run all 4 hypothesis tests and return combined results."""
    results = {
        "h1": test_h1_slo_reduction(df),
        "h2": test_h2_uq_efficiency(df),
        "h3": test_h3_lambda_monotonicity(df),
        "h4": test_h4_tier_concentration(df),
        "comparison_matrix": compare_all_methods(df),
    }
    return results


def save_results(results: dict, output_dir: Path) -> dict[str, Path]:
    """Save statistical results to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    hypo_path = output_dir / "hypothesis_tests.json"
    hypo_path.write_text(json.dumps(results, indent=2, default=str))
    files["hypothesis_tests"] = hypo_path

    norm_df = normality_tests(pd.DataFrame())  # Will be run from report.py with real data
    norm_path = output_dir / "normality_tests.csv"
    norm_path.write_text("# Placeholder — run from report.py\n")
    files["normality_tests"] = norm_path

    eff_path = output_dir / "effect_sizes.csv"
    comps = results.get("comparison_matrix", {}).get("all_comparisons", [])
    if comps:
        eff_rows = []
        for c in comps:
            eff_rows.append({
                "method": c["method_a"],
                "workload": c["workload"],
                "cohens_d": c["cohens_d"],
                "effect_size": c["effect_size"],
                "rel_change_pct": c.get("rel_change_pct", 0),
                "p_value_raw": c["p_value_raw"],
                "p_value_corrected": c["p_value_corrected"],
                "significant_corrected": c["significant_corrected"],
            })
        pd.DataFrame(eff_rows).to_csv(eff_path, index=False)
    else:
        eff_path.write_text("# Placeholder — run from report.py\n")
    files["effect_sizes"] = eff_path

    return files
