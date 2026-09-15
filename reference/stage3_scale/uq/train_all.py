#!/usr/bin/env python3
"""Train all UQ methods (BE, SCP, QR) on all workload patterns."""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_self = Path(__file__).resolve()
_paper3 = _self.parent.parent
if str(_paper3) not in sys.path:
    sys.path.insert(0, str(_paper3))

_repo_root = _paper3.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from predictor.data import load_training_data, NormalizationParams
from uq.bootstrap import BootstrapEnsemble
from uq.conformal import SplitConformal
from uq.quantile import QuantileRegressor
from uq.evaluate import evaluate_uq_method, compare_methods, print_comparison_table

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('uq.train_all')

PATTERN_MAP = {
    'A': ('diurnal', 'pattern_A_24h.csv'),
    'B': ('bursty', 'pattern_B_24h.csv'),
    'C': ('batch_ramp', 'pattern_C_24h.csv'),
    'D': ('signaling', 'pattern_D_24h.csv'),
}

TRAINING_DATA_DIR = _paper3 / 'outputs' / 'training_data'
MODELS_DIR = _paper3 / 'models' / 'uq'
OUTPUTS_DIR = _paper3 / 'outputs' / 'uq_evaluation'


def load_pattern_data(pattern: str) -> tuple:
    """Load training data for a workload pattern."""
    pattern_name, _filename = PATTERN_MAP[pattern]
    # Auto-detect training data file: prefer larger files (more data = better)
    candidates = sorted(
        TRAINING_DATA_DIR.glob(f'pattern_{pattern}_*.csv'),
        key=lambda p: p.stat().st_size, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"No training data found for pattern {pattern} in {TRAINING_DATA_DIR}")
    csv_path = candidates[0]  # Largest file (most data)

    logger.info("Loading %s from %s", pattern_name, csv_path)

    train_loader, val_loader, test_loader, norm = load_training_data(
        str(csv_path), h=60, k=2, batch_size=64, device='cpu'
    )

    X_train_list, y_train_list = [], []
    for Xb, yb in train_loader:
        X_train_list.append(Xb.numpy())
        y_train_list.append(yb.numpy())
    X_train = np.concatenate(X_train_list)  # (N, h, 1)
    y_train = np.concatenate(y_train_list)  # (N, k)

    X_test_list, y_test_list = [], []
    for Xb, yb in test_loader:
        X_test_list.append(Xb.numpy())
        y_test_list.append(yb.numpy())
    X_test = np.concatenate(X_test_list)  # (N, h, 1)
    y_test = np.concatenate(y_test_list)  # (N, k)

    logger.info("  Train: %d samples, Test: %d samples", len(X_train), len(X_test))
    return X_train, y_train, X_test, y_test, norm


def train_and_evaluate(pattern: str, methods: list[str], args) -> dict:
    """Train and evaluate UQ methods on a single pattern."""
    pattern_name = PATTERN_MAP[pattern][0]
    logger.info("=" * 60)
    logger.info("PATTERN %s (%s)", pattern, pattern_name)
    logger.info("=" * 60)

    X_train, y_train, X_test, y_test, norm = load_pattern_data(pattern)

    fitted_methods = {}
    results = {}

    for method_name in methods:
        logger.info("-" * 40)
        logger.info("Training %s on pattern %s...", method_name.upper(), pattern)
        t_start = time.time()

        output_dir = MODELS_DIR / pattern_name / method_name
        eval_dir = OUTPUTS_DIR / pattern_name
        eval_dir.mkdir(parents=True, exist_ok=True)

        try:
            if method_name == 'be':
                uq = BootstrapEnsemble(
                    B=args.be_members,
                    alpha=args.alpha,
                    h=60, k=2, device='cpu',
                    seed=args.seed,
                )
                uq.epochs = args.epochs
                uq.patience = args.patience
                uq.norm_params = norm
                uq.fit((X_train, y_train))

            elif method_name == 'scp':
                uq = SplitConformal(
                    alpha=args.alpha,
                    h=60, k=2, device='cpu',
                )
                uq.norm_params = norm
                uq.fit((X_train, y_train))

            elif method_name == 'qr':
                uq = QuantileRegressor(
                    alpha=args.alpha,
                    h=60, k=2, device='cpu',
                    seed=args.seed,
                )
                uq.epochs = args.epochs
                uq.patience = args.patience
                uq.norm_params = norm
                uq.fit((X_train, y_train))

            uq.save(str(output_dir))
            dt = time.time() - t_start
            logger.info("%s trained + saved in %.1fs", method_name.upper(), dt)

            test_data = (X_test, y_test)
            eval_result = evaluate_uq_method(uq, test_data)
            fitted_methods[method_name] = uq
            results[method_name] = eval_result

            logger.info(
                "  %s: coverage=%.1f%% (target %.0f%%), width=%.1f, "
                "MAPE=%.1f%%, inference=%.2fms",
                method_name.upper(),
                eval_result['empirical_coverage_pct'],
                (1 - args.alpha) * 100,
                eval_result['mean_interval_width'],
                eval_result['mean_point_mape_pct'],
                eval_result['mean_inference_ms'],
            )

            if method_name == 'qr':
                logger.info("  QR crossing rate: %.2f%%",
                             eval_result.get('crossing_rate_pct', 0))

        except Exception as e:
            logger.error("Failed to train %s: %s", method_name, e)
            import traceback
            traceback.print_exc()

    if len(fitted_methods) > 1:
        comparison_path = eval_dir / 'comparison.yaml'
        comparison = compare_methods(fitted_methods, test_data, str(comparison_path))
        print_comparison_table(comparison['methods'])

    summary_path = eval_dir / 'summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    logger.info("Per-pattern results saved to %s", summary_path)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train UQ methods (BE, SCP, QR) on workload patterns'
    )
    pattern_group = parser.add_mutually_exclusive_group(required=True)
    pattern_group.add_argument('--pattern', type=str, choices=['A', 'B', 'C', 'D'],
                               help='Single workload pattern to train on')
    pattern_group.add_argument('--all', action='store_true',
                               help='Train on all 4 workload patterns')

    parser.add_argument('--method', type=str, choices=['be', 'scp', 'qr'],
                        help='Train only a specific method (default: all three)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Confidence level for intervals (default: 0.1 = 90%%)')
    parser.add_argument('--be-members', type=int, default=10,
                        help='Number of ensemble members for BE (default: 10)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation (train + save only)')

    args = parser.parse_args()

    methods = [args.method] if args.method else ['be', 'scp', 'qr']
    patterns = ['A', 'B', 'C', 'D'] if args.all else [args.pattern]

    logger.info("UQ Training: patterns=%s, methods=%s, alpha=%.2f",
                 patterns, methods, args.alpha)
    logger.info("Model output: %s", MODELS_DIR)
    logger.info("Eval output:  %s", OUTPUTS_DIR)

    all_results = {}
    total_start = time.time()

    for pattern in patterns:
        results = train_and_evaluate(pattern, methods, args)
        all_results[pattern] = results

    if args.all and len(patterns) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("CROSS-PATTERN SUMMARY")
        logger.info("=" * 60)

        for method in methods:
            logger.info("\n--- %s ---", method.upper())
            for pattern in patterns:
                if method in all_results.get(pattern, {}):
                    r = all_results[pattern][method]
                    logger.info(
                        "  %s: coverage=%.1f%% (gap=%+.1f), width=%.1f, "
                        "MAPE=%.1f%%, inf=%.2fms",
                        pattern, r['empirical_coverage_pct'],
                        r['coverage_gap_pct'],
                        r['mean_interval_width'],
                        r['mean_point_mape_pct'],
                        r['mean_inference_ms'],
                    )

    total_dt = time.time() - total_start
    logger.info("\nAll done in %.0fs (%.1f min)", total_dt, total_dt / 60)


if __name__ == '__main__':
    main()
