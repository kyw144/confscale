#!/usr/bin/env python3
"""Run the experiment analysis pipeline."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_here = Path(__file__).resolve().parent
_parent = _here.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from analysis.loader import load_runs, load_timeseries, validate_data
from analysis.metrics import (
    aggregate_by_method_workload,
    compute_lambda_sweep,
    compute_slo_violation_rate,
    compute_summary,
    compute_uq_comparison,
)
from analysis.stats import (
    compare_all_methods,
    heterogeneity_tests,
    normality_tests,
    run_all_tests,
    save_results,
)
from analysis.tables import generate_all_tables
from analysis.figures import generate_all_figures

logger = logging.getLogger("report")


def generate_synthetic_runs(output_dir: Path) -> Path:
    """Generate a synthetic experiment dataset for testing the analysis pipeline."""
    METHODS = [
        "hpa-reactive", "hpa-predictive", "hpa-predictive-safety",
        "keda", "base-inspired",
        "confscale-be", "confscale-scp", "confscale-qr",
    ]
    WORKLOADS = ["A", "B", "C", "D"]
    REPLICATES = 3
    DURATION = 3600

    rng = np.random.RandomState(42)

    # Realistic baseline SLO violation rates per workload (from plan estimates)
    baseline_rates = {
        "A": 0.012,  # Diurnal — easy, predictable
        "B": 0.087,  # Bursty — hard
        "C": 0.045,  # Batch-Ramp — moderate
        "D": 0.062,  # Signaling — moderate-hard
    }

    # Method effects (multiplier on baseline, plus noise)
    # Lower = better (fewer violations)
    method_effects = {
        "hpa-reactive": 1.00,
        "hpa-predictive": 1.05,
        "hpa-predictive-safety": 0.80,
        "keda": 0.95,
        "base-inspired": 0.80,
        "confscale-be": 0.69,
        "confscale-scp": 0.60,
        "confscale-qr": 0.63,
    }

    base_overhead = {
        "A": 750,
        "B": 5200,
        "C": 3400,
        "D": 4100,
    }

    method_overhead_mult = {
        "hpa-reactive": 1.00,
        "hpa-predictive": 0.90,
        "hpa-predictive-safety": 1.15,
        "keda": 0.95,
        "base-inspired": 0.85,
        "confscale-be": 0.78,
        "confscale-scp": 0.72,
        "confscale-qr": 0.75,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for method in METHODS:
        for workload in WORKLOADS:
            for rep in range(1, REPLICATES + 1):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_id = f"{method}_{workload.lower()}_rep{rep}_{timestamp}"
                run_dir = output_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                base_rate = baseline_rates[workload]
                effect = method_effects[method]
                noise_factor = 0.15  # 15% noise

                slo_rate = max(0, base_rate * effect * (1 + rng.normal(0, noise_factor)))
                slo_rate = min(slo_rate, 1.0)  # Cap at 100%

                # p95 latency — correlate with violation rate
                p95_base = 80 + slo_rate * 150  # 80ms base, scales with violations
                p95_ms = p95_base + rng.normal(0, 15)

                mean_reps = 3 + rng.uniform(-0.5, 1.0) * (1 + int(workload != "A")) * (1 - effect)
                mean_reps = max(1, mean_reps)

                overhead = base_overhead[workload] * method_overhead_mult[method]
                overhead *= (1 + rng.normal(0, 0.1))
                overhead = max(0, int(overhead))

                metrics = {
                    "run_id": run_id,
                    "method": method,
                    "workload": workload,
                    "replicate": rep,
                    "duration_s": DURATION,
                    "requests": {
                        "total": float(5000 + rng.randint(-200, 200)),
                        "errors": 0.0,
                        "error_rate": 0.0,
                    },
                    "resources": {
                        "mean_replicas": round(mean_reps, 2),
                        "max_replicas": round(mean_reps * 1.5),
                        "replica_churn": float(rng.randint(1, 5)),
                        "overhead_replica_seconds": overhead,
                    },
                    "slo": {
                        "p50_ms": round(p95_ms * 0.4, 2),
                        "p95_ms": round(p95_ms, 2),
                        "p99_ms": round(p95_ms * 1.3, 2),
                        "violation_rate": round(slo_rate, 4),
                        "violation_intervals": int(slo_rate * DURATION / 15),
                    },
                    "e2e": {
                        "p50_ms": round(p95_ms * 0.45, 1),
                        "p95_ms": round(p95_ms * 1.05, 1),
                        "p99_ms": round(p95_ms * 1.4, 1),
                        "slo_violation_rate": round(slo_rate, 4),
                        "slo_violation_intervals": int(slo_rate * DURATION / 15),
                        "total_intervals": DURATION // 15,
                    },
                }

                (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

                config = {
                    "run_id": run_id,
                    "method": method,
                    "workload": workload,
                    "replicate": rep,
                    "duration_s": DURATION,
                    "actual_duration_s": DURATION,
                    "complexity": 50000,
                    "frontend_url": "http://localhost:30080",
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "status": "ok",
                }
                (run_dir / "run_config.yaml").write_text(yaml.dump(config))

                ts_path = run_dir / "timeseries.csv"
                ts_lines = ["timestamp,p95_ms,p50_ms,rps,error_rps,replicas,cpu_cores,violating"]
                ts_lines.append("1000000.0,100.0,50.0,30.0,0.0,3.0,0.1,0")
                ts_path.write_text("\n".join(ts_lines) + "\n")

                rows.append(metrics)

    log_path = output_dir / "run_log.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)

    logger.info("Generated %d synthetic runs in %s", len(rows), output_dir)
    return output_dir


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    figures_only: bool = False,
    tables_only: bool = False,
    synthetic: bool = False,
) -> dict:
    """Run the complete analysis pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if synthetic:
        input_dir = generate_synthetic_runs(output_dir / "synthetic_data")
        output_dir = output_dir / "synthetic_results"
        output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "pipeline_version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }

    logger.info("=" * 60)
    logger.info("PHASE 1: Loading experiment data")
    logger.info("=" * 60)

    df = load_runs(input_dir)
    if len(df) == 0:
        logger.error("No valid runs found in %s", input_dir)
        return {"error": "No data found", "output_dir": str(output_dir)}

    validation = validate_data(df)
    logger.info("Data validation: %d runs, %d methods, %d workloads",
                 validation["n_runs"], validation["n_methods"],
                 validation.get("n_workloads", 0))

    validation_path = output_dir / "data_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, default=str))
    results["validation"] = str(validation_path)

    if validation.get("warnings"):
        for w in validation["warnings"]:
            logger.warning("  ⚠ %s", w)

    df = compute_slo_violation_rate(df)

    if not figures_only and not tables_only:
        logger.info("=" * 60)
        logger.info("PHASE 2: Computing metrics")
        logger.info("=" * 60)

        summary = compute_summary(df)
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        results["summary"] = str(summary_path)
        logger.info("Summary: best method = %s (%.4f)",
                     summary.get("best_method"), summary.get("best_method_rate"))

        agg = aggregate_by_method_workload(df)
        agg_path = output_dir / "data" / "aggregated_metrics.csv"
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(agg_path, index=False)
        results["aggregated_metrics"] = str(agg_path)

        # UQ comparison (joins post-hoc calibration + decision latencies)
        uq = compute_uq_comparison(df, input_dir=input_dir)
        if len(uq) > 0:
            uq_path = output_dir / "data" / "uq_comparison.csv"
            uq.to_csv(uq_path, index=True, index_label="method")
            results["uq_comparison"] = str(uq_path)

    if not figures_only and not tables_only:
        logger.info("=" * 60)
        logger.info("PHASE 3: Statistical analysis")
        logger.info("=" * 60)

        stat_dir = output_dir / "statistics"

        all_tests = run_all_tests(df)
        test_path = stat_dir / "hypothesis_tests.json"
        stat_dir.mkdir(parents=True, exist_ok=True)
        test_path.write_text(json.dumps(all_tests, indent=2, default=str))
        results["hypothesis_tests"] = str(test_path)

        for h in ["h1", "h2", "h3", "h4"]:
            h_data = all_tests.get(h, {})
            verdict = h_data.get("verdict", h_data.get("status", "unknown"))
            logger.info("  %s: %s", h.upper(), verdict)

        norm_df = normality_tests(df)
        norm_path = stat_dir / "normality_tests.csv"
        norm_df.to_csv(norm_path, index=False)
        results["normality_tests"] = str(norm_path)

        het_df = heterogeneity_tests(df)
        het_path = stat_dir / "heterogeneity_tests.csv"
        het_df.to_csv(het_path, index=False)

        comparisons = compare_all_methods(df)
        comp_path = stat_dir / "comparison_matrix.json"
        comp_path.write_text(json.dumps(comparisons, indent=2, default=str))
        results["comparison_matrix"] = str(comp_path)

        eff_path = stat_dir / "effect_sizes.csv"
        all_comps = comparisons.get("all_comparisons", [])
        if all_comps:
            eff_rows = [{
                "method": c["method_a"],
                "workload": c["workload"],
                "cohens_d": c["cohens_d"],
                "effect_size": c["effect_size"],
                "rel_change_pct": c.get("rel_change_pct", 0),
                "p_value_raw": c["p_value_raw"],
                "p_value_corrected": c["p_value_corrected"],
                "significant_corrected": c["significant_corrected"],
            } for c in all_comps]
            pd.DataFrame(eff_rows).to_csv(eff_path, index=False)
            results["effect_sizes"] = str(eff_path)

    if not figures_only:
        logger.info("=" * 60)
        logger.info("PHASE 4: Generating tables")
        logger.info("=" * 60)

        tables_dir = output_dir / "tables"
        all_tables = generate_all_tables(df, tables_dir, input_dir=input_dir)

        for name, paths in all_tables.items():
            for fmt, path in paths.items():
                logger.info("  %s (%s): %s", name, fmt, path)
        results["tables"] = {
            name: {fmt: str(p) for fmt, p in paths.items()}
            for name, paths in all_tables.items()
        }

    if not tables_only:
        logger.info("=" * 60)
        logger.info("PHASE 5: Generating figures")
        logger.info("=" * 60)

        figures_dir = output_dir / "figures"
        all_figures = generate_all_figures(df, input_dir, figures_dir)

        for name, path in all_figures.items():
            logger.info("  %s: %s", name, path)
        results["figures"] = {name: str(path) for name, path in all_figures.items()}

    if not figures_only and not tables_only:
        logger.info("=" * 60)
        logger.info("PHASE 6: Generating reproducibility package")
        logger.info("=" * 60)

        readme = f"""# P3 Experiment Results — Reproducibility Package

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Pipeline version: 1.0.0
Input: {input_dir}
Output: {output_dir}

## Contents

- **tables/** — 4 LaTeX + Markdown tables
- **figures/** — Publication-quality PDF figures (architecture, workload patterns, SLO/overhead bars, burst excerpt, time-in-tier). The coverage-calibration figure (`fig_8_*`) is produced by `analysis/calibration.py`; the λ-Pareto figure by `analysis/lambda_analysis.py`.
- **statistics/** — Hypothesis tests, effect sizes, normality diagnostics
- **data/** — Aggregated metrics CSV
- **reproduce.sh** — One-command reproduction script

## Reproduction

```bash
.venv/bin/python src/stage3_scale/analysis/report.py \\
    --input {input_dir} \\
    --output {output_dir}
```

## Data Summary

- Runs analyzed: {len(df)}
- Methods: {', '.join(sorted(df['method'].unique()))}
- Workloads: {', '.join(sorted(df['workload'].unique()))}
- Replicates per cell: {df.groupby(['method', 'workload']).size().min()}

## Citation

This package supports the P3 paper on confidence-aware predictive autoscaling.
"""
        readme_path = output_dir / "README.md"
        readme_path.write_text(readme)

        # Reproduce script — paths resolve relative to this script's location,
        # so it can be invoked from anywhere and survives moves of the repo root.
        reproduce = f"""#!/bin/bash
# P3 Analysis — reproduction script
#
# Re-runs the analysis pipeline against the 124-cell unified matrix.
# Auto-generated by analysis/report.py Phase 6; edits here will be overwritten
# on the next pipeline run — update the template in src/stage3_scale/analysis/report.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${{REPO_ROOT}}/.venv/bin/python"
INPUT_DIR="{input_dir}"
OUTPUT_DIR="{output_dir}"

if [[ ! -x "${{PYTHON}}" ]]; then
    echo "ERROR: ${{PYTHON}} not found." >&2
    echo "Create the env first:" >&2
    echo "    uv venv ${{REPO_ROOT}}/.venv --python 3.12" >&2
    echo "    uv pip install --python ${{PYTHON}} -r ${{REPO_ROOT}}/src/stage3_scale/requirements.txt" >&2
    exit 1
fi

"${{PYTHON}}" "${{REPO_ROOT}}/src/stage3_scale/analysis/report.py" \\
    --input "${{INPUT_DIR}}" \\
    --output "${{OUTPUT_DIR}}"

echo ""
echo "Results: ${{OUTPUT_DIR}}"
echo "  Tables:  ${{OUTPUT_DIR}}/tables/"
echo "  Figures: ${{OUTPUT_DIR}}/figures/"
echo "  Stats:   ${{OUTPUT_DIR}}/statistics/"
"""
        script_path = output_dir / "reproduce.sh"
        script_path.write_text(reproduce)
        script_path.chmod(0o755)

        req_path = output_dir / "requirements-analysis.txt"
        req_path.write_text(
            "pandas>=2.0.0\nnumpy>=1.24.0\nscipy>=1.10.0\n"
            "matplotlib>=3.7.0\nseaborn>=0.12.0\npyyaml>=6.0\n"
        )

        logger.info("  README: %s", readme_path)
        logger.info("  Script: %s", script_path)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Results in: %s", output_dir)
    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="P3 Experiment Analysis Pipeline"
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Root directory with experiment run outputs (e.g., outputs/)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/"),
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Regenerate only figures (skip statistics and tables)",
    )
    parser.add_argument(
        "--tables-only", action="store_true",
        help="Regenerate only tables (skip statistics and figures)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic test data and analyze it",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        figures_only=args.figures_only,
        tables_only=args.tables_only,
        synthetic=args.synthetic,
    )

    if "error" in results:
        logger.error("Pipeline failed: %s", results["error"])
        sys.exit(1)

    print(json.dumps({
        k: v for k, v in results.items()
        if k not in ("figures", "tables") or isinstance(v, str)
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
