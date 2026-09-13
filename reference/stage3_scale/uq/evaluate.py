"""
Cross-Method Evaluation Framework — standardized comparison of UQ methods.

Evaluates all three UQ methods (BE, SCP, QR) on the same test data
using standardized metrics: empirical coverage, interval width, MAPE,
inference time, confidence scores, tier distribution.

This produces the comparison tables used by the analysis pipeline (task 08).
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# Resolve sibling imports
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from . import UncertaintyQuantifier
from predictor.data import NormalizationParams

logger = logging.getLogger(__name__)


def evaluate_uq_method(uq: UncertaintyQuantifier,
                       test_data: tuple) -> dict:
    """Standardized evaluation for a single UQ method.

    Args:
        uq: Fitted UQ method with norm_params set
        test_data: (X_test, y_test) — normalized numpy arrays
                   X_test shape (N, h, 1) or (N, h)
                   y_test shape (N, k)

    Returns:
        dict with aggregate metrics
    """
    X_test, y_test = test_data
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    # Denormalize ground truth
    y_test_rps = np.array([
        uq.norm_params.denormalize(y_test[i])
        for i in range(len(y_test))
    ])

    k = uq.k

    # Accumulators
    coverages = []       # per-step: is y_true in CI?
    interval_widths = []  # per-step: CI width
    point_mapes = []      # per-step: |y - ŷ|/y
    confidence_scores = []
    tiers = {1: 0, 2: 0, 3: 0}
    times_ms = []

    # Per-horizon accumulators
    per_horizon = {
        f'horizon_{step}': {
            'coverage': [],
            'width': [],
            'mape': [],
        }
        for step in range(k)
    }

    crossing_count = 0

    for i in range(len(X_test)):
        # Convert normalized history to raw RPS for predict_with_uncertainty
        history_raw = uq.norm_params.denormalize(X_test[i].flatten())

        t0 = time.time()
        pred = uq.predict_with_uncertainty(history_raw)
        times_ms.append((time.time() - t0) * 1000)

        # Check for crossing quantiles (QR-specific)
        if pred.get('metadata', {}).get('has_crossing', False):
            crossing_count += 1

        y_true = y_test_rps[i]
        for step in range(k):
            in_interval = float(
                pred['ci_lower'][step] <= y_true[step] <= pred['ci_upper'][step]
            )
            coverages.append(in_interval)
            interval_widths.append(float(pred['ci_upper'][step] - pred['ci_lower'][step]))

            denom = y_true[step] + 1e-6
            mape = float(abs(y_true[step] - pred['point_forecast'][step]) / denom)
            point_mapes.append(mape)

            per_horizon[f'horizon_{step}']['coverage'].append(in_interval)
            per_horizon[f'horizon_{step}']['width'].append(
                float(pred['ci_upper'][step] - pred['ci_lower'][step]))
            per_horizon[f'horizon_{step}']['mape'].append(mape)

        confidence_scores.append(pred['confidence_score'])
        tiers[pred['tier']] += 1

    # Aggregate
    result = {
        'method': uq.method,
        'n_samples': len(X_test),
        'alpha': uq.alpha,
        'target_coverage': 1.0 - uq.alpha,

        # Aggregate metrics
        'empirical_coverage_pct': float(np.mean(coverages) * 100),
        'coverage_std': float(np.std(coverages)),
        'coverage_gap_pct': float((np.mean(coverages) - (1.0 - uq.alpha)) * 100),

        'mean_interval_width': float(np.mean(interval_widths)),
        'median_interval_width': float(np.median(interval_widths)),
        'interval_width_std': float(np.std(interval_widths)),

        'mean_point_mape_pct': float(np.mean(point_mapes) * 100),
        'median_point_mape_pct': float(np.median(point_mapes) * 100),

        'mean_confidence_score': float(np.mean(confidence_scores)),
        'median_confidence_score': float(np.median(confidence_scores)),

        'mean_inference_ms': float(np.mean(times_ms)),
        'p95_inference_ms': float(np.percentile(times_ms, 95)),
        'max_inference_ms': float(np.max(times_ms)),

        'tier_distribution': tiers,
        'tier_pct': {
            'high_confidence': tiers[1] / len(X_test) * 100,
            'medium_confidence': tiers[2] / len(X_test) * 100,
            'low_confidence': tiers[3] / len(X_test) * 100,
        },

        # Per-horizon breakdown
        'per_horizon': {
            h_key: {
                'coverage_pct': float(np.mean(vals['coverage']) * 100),
                'mean_width': float(np.mean(vals['width'])),
                'mean_mape_pct': float(np.mean(vals['mape']) * 100),
            }
            for h_key, vals in per_horizon.items()
        },
    }

    # Method-specific diagnostics
    if uq.method == 'qr':
        result['crossing_rate_pct'] = crossing_count / len(X_test) * 100
    elif uq.method == 'be':
        result['ensemble_size'] = getattr(uq, 'B', 0)
    elif uq.method == 'scp':
        result['q_hat'] = getattr(uq, 'q_hat', np.array([])).tolist()

    return result


def compare_methods(methods: dict[str, UncertaintyQuantifier],
                    test_data: tuple,
                    output_path: Optional[str] = None) -> dict:
    """Evaluate multiple UQ methods on the same test data and compare.

    Args:
        methods: dict mapping method name → fitted UQ instance
        test_data: (X_test, y_test) — normalized numpy arrays
        output_path: If provided, save results as YAML

    Returns:
        dict with per-method results and comparison summary
    """
    results = {}
    for name, uq in methods.items():
        logger.info("Evaluating %s...", name)
        results[name] = evaluate_uq_method(uq, test_data)

    # Comparison summary
    summary = {
        'comparison': {
            'best_coverage': min(results.items(),
                                 key=lambda x: abs(x[1]['coverage_gap_pct']))[0],
            'narrowest_intervals': min(results.items(),
                                       key=lambda x: x[1]['mean_interval_width'])[0],
            'best_mape': min(results.items(),
                            key=lambda x: x[1]['mean_point_mape_pct'])[0],
            'fastest_inference': min(results.items(),
                                     key=lambda x: x[1]['mean_inference_ms'])[0],
        },
        'methods': results,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        logger.info("Comparison saved to %s", output_path)

    return summary


def print_comparison_table(results: dict) -> None:
    """Pretty-print a comparison table of UQ methods."""
    header = f"{'Metric':<35} {'BE':>10} {'SCP':>10} {'QR':>10} {'Best':>10}"
    sep = "-" * 75
    lines = [sep, header, sep]

    metrics = [
        ('Coverage (%)', 'empirical_coverage_pct', '{:.1f}'),
        ('Coverage Gap (%)', 'coverage_gap_pct', '{:+.1f}'),
        ('Mean CI Width', 'mean_interval_width', '{:.1f}'),
        ('Point MAPE (%)', 'mean_point_mape_pct', '{:.1f}'),
        ('Confidence Score', 'mean_confidence_score', '{:.2f}'),
        ('Inference (ms)', 'mean_inference_ms', '{:.2f}'),
        ('Tier 1 (High Conf %)', 'tier_pct.high_confidence', '{:.1f}'),
        ('Tier 3 (Low Conf %)', 'tier_pct.low_confidence', '{:.1f}'),
    ]

    for label, key, fmt in metrics:
        vals = {}
        for method in ['be', 'scp', 'qr']:
            if method not in results:
                vals[method] = 'N/A'
                continue
            r = results[method]
            # Handle nested keys like 'tier_pct.high_confidence'
            if '.' in key:
                parts = key.split('.')
                v = r
                for p in parts:
                    v = v.get(p, 0)
            else:
                v = r.get(key, 0)
            vals[method] = fmt.format(v)

        # Determine best
        if all(isinstance(v, str) for v in vals.values()):
            best = '—'
        elif 'coverage_gap' in key:
            best = min(vals.items(), key=lambda x: abs(float(x[1])))[0].upper()
        elif any(w in key for w in ['MAPE', 'Width', 'Confidence', 'ms', 'Low']):
            best = min(vals.items(), key=lambda x: float(x[1]))[0].upper()
        elif 'Coverage' in key or 'Tier 1' in key:
            best = max(vals.items(), key=lambda x: float(x[1]))[0].upper()
        else:
            best = min(vals.items(), key=lambda x: float(x[1]))[0].upper()

        line = f"{label:<35} {vals['be']:>10} {vals['scp']:>10} {vals['qr']:>10} {best:>10}"
        lines.append(line)

    lines.append(sep)
    print("\n".join(lines))
