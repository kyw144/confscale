#!/usr/bin/env python3
"""GRU, safety-margin, error-monitored and quantile autoscaling baselines."""

if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read README.md#cluster-runs; "
                         "local demo: python -m confscale demo")


import argparse
import json
import logging
import signal
import subprocess
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# Resolve imports — same pattern as orchestrator/controller.py
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from predictor import GRUPredictor
from uq.quantile import QuantileRegressor
from orchestrator.coverage_monitor import CoverageMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] baseline-%(mode)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('baseline-controller')


def query_prometheus(url: str, query: str) -> Optional[float]:
    """Query Prometheus for a single scalar value."""
    try:
        full_url = f"{url}/api/v1/query?query={urllib.request.quote(query)}"
        req = urllib.request.Request(full_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if data.get('status') == 'success' and data['data']['result']:
            return float(data['data']['result'][0]['value'][1])
    except Exception as e:
        logger.warning("Prometheus query failed: %s", e)
    return None


def query_prometheus_range(url: str, query: str, lookback_s: int) -> list[float]:
    """Query Prometheus for a range of values."""
    now = time.time()
    start = now - lookback_s
    try:
        full_url = (
            f"{url}/api/v1/query_range"
            f"?query={urllib.request.quote(query)}"
            f"&start={int(start)}&end={int(now)}&step=30s"
        )
        req = urllib.request.Request(full_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if data.get('status') == 'success' and data['data']['result']:
            return [float(v[1]) for v in data['data']['result'][0]['values']]
    except Exception as e:
        logger.warning("Prometheus range query failed: %s", e)
    return []


def kubectl_scale(context: str, namespace: str, deployment: str, replicas: int):
    """Scale a deployment via kubectl."""
    cmd = [
        'kubectl', f'--context={context}',
        'scale', 'deployment', deployment,
        f'--replicas={replicas}',
        '-n', namespace,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning("kubectl scale failed: %s", result.stderr.strip())
        else:
            logger.debug("Scaled %s to %d", deployment, replicas)
    except Exception as e:
        logger.error("kubectl scale error: %s", e)


def compute_replicas_predicted(
    r_hat: float,
    slo_capacity: float,
    target_util: float,
    min_replicas: int,
    max_replicas: int,
) -> int:
    """Simple: ceil(forecast / (capacity * util))."""
    effective_cap = slo_capacity * target_util
    if effective_cap <= 0:
        return min_replicas
    replicas = int(np.ceil(r_hat / effective_cap))
    return max(min_replicas, min(replicas, max_replicas))


class BurstDetector:
    """Binary burst detector with hysteresis."""

    def __init__(
        self,
        burst_threshold: float = 2.0,
        history_window: int = 5,
        hysteresis_intervals: int = 3,
    ):
        self.burst_threshold = burst_threshold
        self.history_window = history_window
        self.hysteresis_intervals = hysteresis_intervals
        self.in_burst = False
        self.burst_counter = 0

    def update(self, history: list[float]) -> bool:
        """Return True if we're currently in burst mode."""
        if len(history) < 2:
            return self.in_burst

        recent_avg = np.mean(history[-self.history_window:])
        current = history[-1]

        was_in_burst = self.in_burst

        if recent_avg > 0 and current > self.burst_threshold * recent_avg:
            self.in_burst = True
            self.burst_counter = self.hysteresis_intervals
        elif self.in_burst and self.burst_counter > 0:
            # Still in hysteresis window — stay in burst mode
            self.burst_counter -= 1
        else:
            self.in_burst = False

        if self.in_burst and not was_in_burst:
            logger.info("Burst detected (rps=%.1f, avg=%.1f, ratio=%.2f)",
                         current, recent_avg, current / max(recent_avg, 0.001))

        return self.in_burst


class ErrorMonitor:
    """Trailing residual-MAE monitor for the error-monitored baseline."""

    def __init__(self, window: int = 20):
        if window <= 0:
            raise ValueError(f"ErrorMonitor window must be positive, got {window}")
        self.window = window
        self.residuals: deque[float] = deque(maxlen=window)

    def record(self, forecast: float, observed: float) -> float:
        """Append ``|forecast - observed|`` to the window and return it."""
        residual = abs(float(forecast) - float(observed))
        self.residuals.append(residual)
        return residual

    def mae(self) -> float:
        """Mean absolute residual over the current window; 0.0 if empty."""
        if not self.residuals:
            return 0.0
        return float(np.mean(self.residuals))

    def is_elevated(self, threshold: float) -> bool:
        """True when current MAE exceeds the threshold."""
        if not self.residuals:
            return False
        return self.mae() > threshold

    def samples(self) -> int:
        return len(self.residuals)


def build_operator_metrics_summary(
    coverage_monitor: Optional['CoverageMonitor'],
    scale_log: list,
    mode: str,
    final_replicas: int,
) -> Optional[dict]:
    """Build the operator_metrics_summary.json payload."""
    if coverage_monitor is None or not scale_log:
        return None

    total_validated = coverage_monitor.lifetime_validated
    total_covered = coverage_monitor.lifetime_covered
    coverage_rate = (
        total_covered / total_validated if total_validated > 0 else None
    )
    total_scale_ops = sum(
        1 for i in range(1, len(scale_log))
        if scale_log[i]['target_replicas'] != scale_log[i - 1]['target_replicas']
    )
    return {
        'mode': mode,
        'total_intervals': len(scale_log),
        'total_scale_operations': total_scale_ops,
        'final_replicas': final_replicas,
        'coverage_monitor': {
            'enabled': True,
            'final_trailing_coverage': round(
                float(coverage_monitor.trailing_coverage), 4),
            'final_alert_state': int(coverage_monitor.alert_state),
            'total_validated': total_validated,
            'total_covered': total_covered,
            'coverage_rate': (round(coverage_rate, 4)
                              if coverage_rate is not None else None),
            'target_coverage': float(coverage_monitor.target_coverage),
            'window_size': int(coverage_monitor.window_size),
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Baseline scaling controller')
    parser.add_argument('--mode', required=True,
                        choices=['predictive', 'predictive-safety', 'base-inspired',
                                 'error-monitored', 'hpa-qr-monitored'])
    parser.add_argument('--model-dir', required=True,
                        help='Path to predictor model directory. For GRU-backed '
                             'modes (predictive, predictive-safety, base-inspired, '
                             'error-monitored) this is the per-pattern GRU dir. For '
                             'hpa-qr-monitored it is the per-pattern QR dir '
                             '(models/uq/<pattern>/qr/).')
    parser.add_argument('--namespace', default='infosys-benchmark')
    parser.add_argument('--deployment', default='compute-worker')
    parser.add_argument('--prometheus-url', default='http://localhost:9090')
    parser.add_argument('--rps-query',
                        default='sum(rate(frontend_latency_seconds_count[30s]))',
                        help='PromQL query for current RPS. Default measures rate of '
                             'frontend Histogram observations — same metric the '
                             'orchestrator collector uses, so controller input matches '
                             'recorded data.')
    parser.add_argument('--slo-capacity', type=float, default=10.0)
    parser.add_argument('--target-util', type=float, default=0.7)
    parser.add_argument('--safety-margin', type=float, default=0.2,
                        help='Safety margin for predictive-safety mode')
    parser.add_argument('--burst-multiplier', type=float, default=1.5,
                        help='Multiplier for BASE-inspired burst mode')
    parser.add_argument('--burst-threshold', type=float, default=2.0,
                        help='Rate-of-change threshold for burst detection')
    parser.add_argument('--error-window', type=int, default=20,
                        help='Sliding-window size (iterations) for residual MAE '
                             'in error-monitored mode')
    parser.add_argument('--error-threshold', type=float, default=5.0,
                        help='MAE threshold (RPS) above which error-monitored mode '
                             'escalates to conservative scaling')
    parser.add_argument('--error-safety-multiplier', type=float, default=1.5,
                        help='Multiplier applied to the point forecast when '
                             'error-monitored mode is elevated')
    parser.add_argument('--coverage-monitor', action='store_true',
                        help='Enable the online trailing-coverage monitor. '
                             'Validates the prior iteration h=0 prediction interval '
                             'against this iteration\'s RPS and emits '
                             'operator_metrics_summary.json. Forced on for '
                             'hpa-qr-monitored mode.')
    parser.add_argument('--coverage-target', type=float, default=0.90,
                        help='Target trailing coverage for the monitor (default 0.90).')
    parser.add_argument('--coverage-window', type=int, default=30,
                        help='Sliding-window size (iterations) for trailing coverage '
                             '(default 30).')
    parser.add_argument('--min-replicas', type=int, default=1)
    parser.add_argument('--max-replicas', type=int, default=20)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--duration', type=int, default=3600)
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--history-length', type=int, default=60,
                        help='History window size (h) for predictor')
    parser.add_argument('--context', default='kind-confscale-experiments',
                        help='Kube context name')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['start_time'] = datetime.now(timezone.utc).isoformat()
    config['controller_type'] = f'baseline-{args.mode}'
    with open(output_dir / 'controller_config.yaml', 'w') as f:
        yaml.dump(config, f)

    log_extra = {'mode': args.mode}
    logger.info("Starting baseline controller: %s", args.mode,
                extra=log_extra)
    logger.info("  Model: %s", args.model_dir, extra=log_extra)
    logger.info("  Deployment: %s/%s", args.namespace, args.deployment,
                extra=log_extra)
    logger.info("  Duration: %ds, Interval: %ds", args.duration, args.interval,
                extra=log_extra)

    qr_predictor: Optional[QuantileRegressor] = None
    predictor: Optional[GRUPredictor] = None
    if args.mode == 'hpa-qr-monitored':
        try:
            qr_predictor = QuantileRegressor.load(args.model_dir, device='cpu')
            predictor_h = qr_predictor.h
            predictor_k = qr_predictor.k
            logger.info("Loaded QR predictor (h=%d, k=%d, quantiles=%s)",
                        predictor_h, predictor_k, qr_predictor.quantiles,
                        extra=log_extra)
        except Exception as e:
            logger.error("Failed to load QR predictor: %s", e, extra=log_extra)
            sys.exit(1)
    else:
        try:
            predictor = GRUPredictor(args.model_dir, device='cpu')
            predictor_h = predictor.h
            predictor_k = predictor.k
            logger.info("Loaded GRU predictor (h=%d, k=%d)",
                        predictor_h, predictor_k, extra=log_extra)
        except Exception as e:
            logger.error("Failed to load GRU predictor: %s", e, extra=log_extra)
            sys.exit(1)

    burst_detector = None
    if args.mode == 'base-inspired':
        burst_detector = BurstDetector(
            burst_threshold=args.burst_threshold,
        )

    # Score the previous forecast against the current observation.
    error_monitor = None
    prev_forecast: Optional[float] = None
    if args.mode == 'error-monitored':
        error_monitor = ErrorMonitor(window=args.error_window)
        logger.info("ErrorMonitor: window=%d, threshold=%.2f, safety=%.2f",
                    args.error_window, args.error_threshold,
                    args.error_safety_multiplier, extra=log_extra)

    coverage_monitor: Optional[CoverageMonitor] = None
    coverage_enabled = args.coverage_monitor or args.mode == 'hpa-qr-monitored'
    if coverage_enabled:
        coverage_monitor = CoverageMonitor(
            window_size=args.coverage_window,
            target_coverage=args.coverage_target,
        )
        logger.info("CoverageMonitor enabled: target=%.2f, window=%d",
                    args.coverage_target, args.coverage_window, extra=log_extra)

    current_replicas = args.min_replicas
    scale_log = []

    kubectl_scale(args.context, args.namespace, args.deployment, current_replicas)
    time.sleep(5)

    shutdown = False

    def handle_signal(sig, frame):
        nonlocal shutdown
        logger.info("Received signal %s, shutting down...", sig, extra=log_extra)
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    start_time = time.time()
    end_time = start_time + args.duration

    logger.info("Control loop starting. Will run for %ds.", args.duration,
                extra=log_extra)

    while time.time() < end_time and not shutdown:
        loop_start = time.time()

        rps = query_prometheus(args.prometheus_url, args.rps_query)

        history = query_prometheus_range(
            args.prometheus_url, args.rps_query,
            lookback_s=args.history_length * 30,
        )

        # Pad short history with the latest observation so cold starts can produce forecasts.
        seeded = False
        if 0 < len(history) < predictor_h:
            pad_value = history[-1]
            history = [pad_value] * (predictor_h - len(history)) + history
            seeded = True

        # 3. Predict. QR for hpa-qr-monitored, GRU point-forecast otherwise.
        prediction_made = False
        r_hat = 0.0
        forecast = []
        ci_lower_h0: Optional[float] = None
        ci_upper_h0: Optional[float] = None
        in_burst = False
        policy = 'precise'

        if len(history) >= predictor_h:
            try:
                history_array = np.array(history[-predictor_h:], dtype=np.float32)
                if qr_predictor is not None:
                    pred = qr_predictor.predict_with_uncertainty(history_array)
                    r_hat = float(pred['point_forecast'][0])
                    forecast = pred['point_forecast'].tolist()
                    ci_lower_h0 = float(pred['ci_lower'][0])
                    ci_upper_h0 = float(pred['ci_upper'][0])
                else:
                    pred = predictor.predict(history_array)
                    r_hat = float(pred['point_forecast'][0])
                    forecast = pred['point_forecast'].tolist()
                prediction_made = True
            except Exception as e:
                logger.warning("Prediction failed: %s", e, extra=log_extra)
                pred = None

        # Validate the previous interval before recording the next one.
        coverage_state = None
        if coverage_monitor is not None:
            if rps is not None:
                coverage_monitor.validate_pending(float(rps))
            if (prediction_made and ci_lower_h0 is not None
                    and ci_upper_h0 is not None):
                coverage_monitor.record_prediction(ci_lower_h0, ci_upper_h0)
            coverage_state = coverage_monitor.get_state()

        # Error-monitor bookkeeping for the log entry (populated below
        # for error-monitored mode; left as None for other modes).
        error_residual: Optional[float] = None
        error_mae: Optional[float] = None
        error_elevated: bool = False

        if prediction_made:
            if args.mode == 'predictive':
                r_target = r_hat
                policy = 'predictive'

            elif args.mode == 'predictive-safety':
                r_target = r_hat * (1.0 + args.safety_margin)
                policy = 'predictive+safety'

            elif args.mode == 'base-inspired':
                burst_detector.update(history)
                in_burst = burst_detector.in_burst
                if in_burst:
                    r_target = r_hat * args.burst_multiplier
                    policy = 'burst-conservative'
                else:
                    r_target = r_hat
                    policy = 'precise'

            elif args.mode == 'error-monitored':
                # Score the previous forecast against this iteration's
                # observed RPS, then act on the trailing MAE.
                if prev_forecast is not None and rps is not None:
                    error_residual = error_monitor.record(prev_forecast, rps)
                error_mae = error_monitor.mae()
                error_elevated = error_monitor.is_elevated(args.error_threshold)
                if error_elevated:
                    r_target = r_hat * args.error_safety_multiplier
                    policy = 'error-conservative'
                else:
                    r_target = r_hat
                    policy = 'error-precise'

            elif args.mode == 'hpa-qr-monitored':
                # Use the QR upper quantile directly, with the point forecast as fallback.
                r_target = ci_upper_h0 if ci_upper_h0 is not None else r_hat
                policy = 'qr-upper-quantile'

            target_replicas = compute_replicas_predicted(
                r_target,
                slo_capacity=args.slo_capacity,
                target_util=args.target_util,
                min_replicas=args.min_replicas,
                max_replicas=args.max_replicas,
            )

            # Save the forecast after planning so a failed decision cannot affect the next residual.
            if args.mode == 'error-monitored':
                prev_forecast = r_hat
        else:
            # No prediction available — hold at current replicas
            target_replicas = current_replicas
            policy = 'no-prediction'
            r_target = 0.0
            if args.mode == 'error-monitored' and error_monitor is not None:
                error_mae = error_monitor.mae()

        # 4. Apply scaling (no hysteresis for baselines — they are simple)
        if target_replicas != current_replicas:
            old_replicas = current_replicas
            current_replicas = target_replicas
            kubectl_scale(args.context, args.namespace, args.deployment,
                         current_replicas)

        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'elapsed_s': round(time.time() - start_time, 1),
            'rps': rps,
            'forecast': forecast,
            'r_target': round(r_target, 2),
            'policy': policy,
            'in_burst': in_burst,
            'target_replicas': target_replicas,
            'actual_replicas': current_replicas,
            'history_len': len(history),
            'seeded': seeded,
            'mode': args.mode,
        }
        if ci_lower_h0 is not None and ci_upper_h0 is not None:
            log_entry['ci_lower_h0'] = round(ci_lower_h0, 4)
            log_entry['ci_upper_h0'] = round(ci_upper_h0, 4)
        if args.mode == 'error-monitored':
            log_entry['error_monitor'] = {
                'residual': (round(error_residual, 4)
                             if error_residual is not None else None),
                'trailing_mae': (round(error_mae, 4)
                                 if error_mae is not None else None),
                'threshold': args.error_threshold,
                'elevated': error_elevated,
                'samples': error_monitor.samples() if error_monitor else 0,
                'window_size': args.error_window,
                'safety_multiplier': args.error_safety_multiplier,
            }
        if coverage_state is not None:
            log_entry['coverage'] = coverage_state
        scale_log.append(log_entry)

        if prediction_made:
            if args.mode == 'error-monitored':
                logger.info(
                    "t=%4ds | rps=%.1f | forecast=%.1f | r_target=%.1f | "
                    "policy=%s | mae=%.2f (n=%d) | target=%d | cur=%d",
                    log_entry['elapsed_s'], rps or 0,
                    r_hat, r_target, policy,
                    error_mae if error_mae is not None else 0.0,
                    error_monitor.samples() if error_monitor else 0,
                    target_replicas, current_replicas,
                    extra=log_extra,
                )
            else:
                logger.info(
                    "t=%4ds | rps=%.1f | forecast=%.1f | r_target=%.1f | "
                    "policy=%s | target=%d | cur=%d",
                    log_entry['elapsed_s'], rps or 0,
                    r_hat, r_target, policy, target_replicas, current_replicas,
                    extra=log_extra,
                )
        else:
            logger.info("t=%4ds | no prediction yet (history=%d)",
                         log_entry['elapsed_s'], len(history),
                         extra=log_extra)

        # Sleep in short chunks so SIGTERM can flush logs before reset escalates to SIGKILL.
        elapsed = time.time() - loop_start
        sleep_end = time.time() + max(0, args.interval - elapsed)
        while time.time() < sleep_end and not shutdown:
            time.sleep(min(1.0, sleep_end - time.time()))

    # Save scale log. Wrapped in try/except so a flake here still lets
    # the process exit cleanly — better partial data than no data.
    log_path = output_dir / 'controller_scale_log.json'
    try:
        with open(log_path, 'w') as f:
            json.dump(scale_log, f, indent=2)
        logger.info("Scale log saved: %s (%d entries)", log_path, len(scale_log),
                    extra=log_extra)
    except Exception as e:
        logger.error("Failed to save scale log to %s: %s", log_path, e,
                     extra=log_extra)

    metrics_summary = build_operator_metrics_summary(
        coverage_monitor=coverage_monitor,
        scale_log=scale_log,
        mode=args.mode,
        final_replicas=current_replicas,
    )
    if metrics_summary is not None:
        summary_path = output_dir / 'operator_metrics_summary.json'
        try:
            with open(summary_path, 'w') as f:
                json.dump(metrics_summary, f, indent=2)
            logger.info("Metrics summary: %s", json.dumps(metrics_summary),
                        extra=log_extra)
        except Exception as e:
            logger.error("Failed to save metrics summary to %s: %s",
                         summary_path, e, extra=log_extra)

    logger.info("Controller finished. Final replicas: %d, policy: %s",
                 current_replicas, policy, extra=log_extra)


if __name__ == '__main__':
    main()
