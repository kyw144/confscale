#!/usr/bin/env python3
"""Predictive Controller — background subprocess for confidence-aware scaling."""

if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read README.md#cluster-runs; "
                         "local demo: python -m confscale demo")


import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import urllib.request
import yaml

from prometheus_client import Gauge, Histogram, start_http_server, REGISTRY

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from predictor import GRUPredictor
from uq.bootstrap import BootstrapEnsemble
from uq.conformal import SplitConformal
from uq.quantile import QuantileRegressor
from uq.conformal_pid import ConformalPID, EmptyResidualBufferError
from uq.aci import ACI
from baselines.escalation_ladder import EscalationLadder
from orchestrator.coverage_monitor import CoverageMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] controller: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('controller')


METRICS_PREFIX = "confidence_scaler"

g_predicted_rps = Gauge(
    f'{METRICS_PREFIX}_predicted_rps',
    'Point forecast RPS (max over forecast horizon)',
)
g_ci_lower = Gauge(
    f'{METRICS_PREFIX}_ci_lower',
    'CI lower bound RPS',
)
g_ci_upper = Gauge(
    f'{METRICS_PREFIX}_ci_upper',
    'CI upper bound RPS',
)
g_current_tier = Gauge(
    f'{METRICS_PREFIX}_tier',
    'Current confidence tier (1-3)',
)
g_desired_replicas = Gauge(
    f'{METRICS_PREFIX}_desired_replicas',
    'Desired replica count',
)
g_confidence_score = Gauge(
    f'{METRICS_PREFIX}_confidence_score',
    'Confidence score (higher = more uncertain)',
)
g_actual_replicas = Gauge(
    f'{METRICS_PREFIX}_actual_replicas',
    'Current deployment replicas',
)

g_trailing_coverage = Gauge(
    f'{METRICS_PREFIX}_trailing_coverage',
    'Trailing empirical CI coverage at h=0 over the validation window',
)
g_coverage_shortfall = Gauge(
    f'{METRICS_PREFIX}_coverage_shortfall',
    'max(0, target_coverage - trailing_coverage)',
)
g_coverage_alert = Gauge(
    f'{METRICS_PREFIX}_coverage_alert',
    'Coverage alert band: 0 nominal, 1 warning, 2 critical',
)

g_conformal_pid_alpha = Gauge(
    f'{METRICS_PREFIX}_conformal_pid_alpha',
    'Current PID/ACI working miscoverage level α',
)
g_conformal_pid_integral = Gauge(
    f'{METRICS_PREFIX}_conformal_pid_integral',
    'Running sum of signed coverage errors (PID integral term)',
)
g_conformal_pid_derivative = Gauge(
    f'{METRICS_PREFIX}_conformal_pid_derivative',
    'Last-step Δerror (PID derivative term)',
)
g_conformal_pid_quantile_halfwidth = Gauge(
    f'{METRICS_PREFIX}_conformal_pid_quantile_halfwidth',
    'Recalibrator quantile half-width in normalized residual space',
)
g_escalation_ladder_level = Gauge(
    f'{METRICS_PREFIX}_escalation_ladder_level',
    'Coverage-conditional escalation ladder level (0 nominal, 1 widening, 2 conservative)',
)

h_decision_latency = Histogram(
    f'{METRICS_PREFIX}_decision_latency_seconds',
    'Time to compute scaling decision',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)


# Higher confidence_score means greater uncertainty.
TIER_HIGH_CONF = 0.15
TIER_MED_CONF = 0.4

TIER_DOWNGRADE_DELAY = 5      # Intervals before allowing tier downgrade
SCALE_UP_COOLDOWN_S = 30       # Minimum interval between scale-ups
SCALE_DOWN_COOLDOWN_T1_S = 60  # Tier 1 scale-down cooldown
SCALE_DOWN_COOLDOWN_T2_S = 90  # Tier 2 scale-down cooldown
SCALE_DOWN_COOLDOWN_T3_S = 120 # Tier 3 scale-down cooldown


def _tier_cooldown(tier: int) -> float:
    if tier == 3:
        return SCALE_DOWN_COOLDOWN_T3_S
    elif tier == 2:
        return SCALE_DOWN_COOLDOWN_T2_S
    return SCALE_DOWN_COOLDOWN_T1_S


def compute_target_replicas(
    point_forecast: np.ndarray,
    confidence_score: float,
    ci_upper: np.ndarray = None,
    slo_capacity: float = 10.0,
    target_util: float = 0.7,
    safety_factor: float = 2.0,
    min_replicas: int = 1,
    max_replicas: int = 20,
    policy: str = "tier",
    lambda_risk: Optional[float] = None,
) -> tuple[int, int]:
    """Return (tier, replicas) from the maximum horizon demand.

    lambda_risk overrides policy; ci-upper returns tier 0 while hysteresis tracks confidence tiers.
    """
    # Always classify tier so hysteresis cooldown selection works.
    if confidence_score < TIER_HIGH_CONF:
        classified_tier = 1
    elif confidence_score < TIER_MED_CONF:
        classified_tier = 2
    else:
        classified_tier = 3

    if lambda_risk is not None:
        upper = ci_upper if ci_upper is not None else point_forecast * 1.5
        upper_gap = np.maximum(upper - point_forecast, 0.0)
        effective = point_forecast + lambda_risk * upper_gap
        effective_rate = float(np.max(effective))
        tier = classified_tier
    elif policy == "ci-upper":
        if ci_upper is not None:
            effective_rate = float(np.max(ci_upper))
        else:
            effective_rate = float(np.max(point_forecast * 1.5))
        tier = 0  # sentinel: no tier policy active
    else:
        # Default 3-tier policy
        tier = classified_tier
        if tier == 1:
            effective_rate = float(np.max(point_forecast))
        elif tier == 2:
            if ci_upper is not None:
                std_est = (ci_upper - point_forecast) / 1.645  # 90% CI: z=1.645
            else:
                std_est = point_forecast * confidence_score
            effective_rate = float(np.max(point_forecast + safety_factor * std_est))
        else:
            if ci_upper is not None:
                effective_rate = float(np.max(ci_upper))
            else:
                effective_rate = float(np.max(point_forecast * 1.5))

    replicas = int(np.ceil(effective_rate / (slo_capacity * target_util)))
    replicas = max(min_replicas, min(replicas, max_replicas))

    return tier, replicas


class HysteresisManager:
    """Prevent oscillation with tier-downgrade delays and tier-specific cooldowns."""

    def __init__(self):
        self.current_tier: int = 1
        self.tier_duration: int = 0         # Consecutive intervals at current tier
        self.proposed_downgrade_count: int = 0  # Intervals proposing downgrade
        self.last_scale_up_time: float = 0.0
        self.last_scale_down_time: float = 0.0
        self.scale_up_cooldown_s: float = SCALE_UP_COOLDOWN_S

    def apply(self, proposed_tier: int, proposed_replicas: int,
              current_replicas: int) -> tuple[int, int]:
        """Return (tier, replicas) after downgrade persistence and scale-down cooldowns."""
        now = time.time()

        if proposed_tier < self.current_tier:
            self.proposed_downgrade_count += 1
            if self.proposed_downgrade_count < TIER_DOWNGRADE_DELAY:
                proposed_tier = self.current_tier
            else:
                # Allow downgrade after delay
                self.proposed_downgrade_count = 0
        elif proposed_tier > self.current_tier:
            self.proposed_downgrade_count = 0
        else:
            # Same tier
            self.proposed_downgrade_count = 0

        self.current_tier = proposed_tier

        final_replicas = proposed_replicas

        if proposed_replicas > current_replicas:
            # Scale-up: enforce cooldown
            if self.last_scale_up_time > 0 and (now - self.last_scale_up_time) < self.scale_up_cooldown_s:
                final_replicas = current_replicas
            else:
                self.last_scale_up_time = now
        elif proposed_replicas < current_replicas:
            # Scale-down: enforce tier-specific cooldown
            cooldown_s = _tier_cooldown(self.current_tier)
            if self.last_scale_down_time > 0 and (now - self.last_scale_down_time) < cooldown_s:
                final_replicas = current_replicas
            else:
                self.last_scale_down_time = now

        return self.current_tier, final_replicas


def query_prometheus(url: str, query: str) -> Optional[float]:
    """Query Prometheus for a single scalar value."""
    try:
        full_url = f"{url}/api/v1/query?query={urllib.request.quote(query)}"
        req = urllib.request.Request(full_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if data.get('status') == 'success' and data['data']['result']:
            value = float(data['data']['result'][0]['value'][1])
            return value
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
            values = [float(v[1]) for v in data['data']['result'][0]['values']]
            return values
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
            raise RuntimeError(f"kubectl scale failed: {result.stderr.strip()}")
        else:
            logger.debug("Scaled %s to %d", deployment, replicas)
    except Exception as e:
        logger.error("kubectl scale error: %s", e)
        raise


def kubectl_get_replicas(context: str, namespace: str, deployment: str) -> Optional[int]:
    """Get observed ready replica count of a deployment (not its requested spec)."""
    cmd = [
        'kubectl', f'--context={context}',
        'get', 'deployment', deployment,
        '-n', namespace,
        '-o', 'jsonpath={.status.readyReplicas}',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return int(result.stdout.strip() or 0)
    except Exception as e:
        logger.warning("kubectl get replicas failed: %s", e)
    return None


def _update_prometheus_metrics(prediction_made: bool, pred: dict,
                                tier: int, target_replicas: int,
                                current_replicas: int, latency_s: float):
    if prediction_made and pred:
        g_predicted_rps.set(float(np.max(pred.get('point_forecast', [0.0]))))
        g_ci_lower.set(float(np.max(pred.get('ci_lower', [0.0]))))
        g_ci_upper.set(float(np.max(pred.get('ci_upper', [0.0]))))
        g_confidence_score.set(float(pred.get('confidence_score', 0.0)))
    else:
        g_predicted_rps.set(0.0)
        g_ci_lower.set(0.0)
        g_ci_upper.set(0.0)
        g_confidence_score.set(0.0)

    g_current_tier.set(tier)
    g_desired_replicas.set(target_replicas)
    g_actual_replicas.set(current_replicas)
    h_decision_latency.observe(latency_s)


def main():
    parser = argparse.ArgumentParser(description='Confidence-aware scaling controller')
    parser.add_argument('--method', required=True, choices=['be', 'scp', 'qr'])
    parser.add_argument('--model-dir', required=True, help='Path to UQ model directory')
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
    parser.add_argument('--safety-factor', type=float, default=2.0)
    parser.add_argument('--min-replicas', type=int, default=1)
    parser.add_argument('--max-replicas', type=int, default=20)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--duration', type=int, default=3600)
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--hysteresis', type=int, default=1,
                       help='Min replica change to trigger scaling')
    parser.add_argument('--history-length', type=int, default=60,
                       help='History window size (h) for predictor')
    parser.add_argument('--context', default='kind-confscale-experiments',
                       help='Kube context name')
    parser.add_argument('--metrics-port', type=int, default=int(os.environ.get('CONFSCALE_METRICS_PORT', '9091')),
                       help='Port for Prometheus /metrics endpoint '
                            '(default 9091 to avoid conflict with Prometheus on 9090)')
    parser.add_argument('--policy', choices=['tier', 'ci-upper'], default='tier',
                       help="'tier' = 3-tier confidence policy (default); "
                            "'ci-upper' = always scale to ci_upper (MagicScaler-style baseline)")
    parser.add_argument('--lambda-risk', type=float, default=None,
                       help='Continuous risk weight [0,1] blending point and ci_upper. '
                            'Overrides --policy when set.')
    parser.add_argument('--online-recal', action='store_true',
                       help='SCP only: incrementally recalibrate q_hat from streaming '
                            'residuals. Each iteration observes prior predictions to '
                            'build per-horizon residual buffers.')
    parser.add_argument('--recal-buffer-size', type=int, default=200,
                       help='Max residuals retained per horizon for online recal (FIFO)')
    parser.add_argument('--recal-warmup', type=int, default=30,
                       help='Minimum residuals per horizon before q_hat updates begin')
    parser.add_argument('--recal-warmstart', action='store_true',
                       help='SCP rolling-origin (T7b): pre-seed the per-horizon residual '
                            'buffers from the offline q_hat so online recalibration begins at '
                            'decision 0 instead of after --recal-warmup cold-start decisions. '
                            'Each buffer is seeded with --recal-warmstart-n half-normal residuals '
                            'whose (1-alpha) order statistic is pinned to the offline q_hat[h] '
                            '(reproduces the offline interval at t=0; additive, default off).')
    parser.add_argument('--recal-warmstart-n', type=int, default=None,
                       help='Warm-start seed count per horizon (default: --recal-warmup).')
    parser.add_argument('--recal-warmstart-seed', type=int, default=0,
                       help='RNG seed for the warm-start reconstruction (reproducibility).')
    parser.add_argument('--coverage-monitor', action='store_true',
                       help='Enable online empirical-coverage monitor (Analyze side). '
                            'Tracks h=0 CI coverage over a trailing window and exports '
                            'trailing_coverage / coverage_shortfall / coverage_alert as '
                            'Prometheus gauges and as a `coverage` block in the scale log. '
                            'No Plan-side action is taken — that wiring is C2-Plan.')
    parser.add_argument('--coverage-target', type=float, default=0.90,
                       help='Nominal coverage target T used by the coverage monitor. '
                            'Alert tips to warning when trailing < T and to critical when '
                            'trailing < 0.85·T.')
    parser.add_argument('--coverage-window', type=int, default=30,
                       help='Sliding-window size (validations) for trailing coverage.')
    parser.add_argument('--recalibrator',
                       choices=['none', 'rolling-origin', 'aci', 'pid'],
                       default='none',
                       help='Online recalibrator (SCP only). Replaces --online-recal '
                            '("rolling-origin" is the same behavior under the new name).')
    parser.add_argument('--pid-k-p', type=float, default=0.1)
    parser.add_argument('--pid-k-i', type=float, default=0.01)
    parser.add_argument('--pid-k-d', type=float, default=0.05)
    parser.add_argument('--aci-eta', type=float, default=0.1)
    parser.add_argument('--recal-alpha-clip-low', type=float, default=1e-4)
    parser.add_argument('--recal-alpha-clip-high', type=float, default=0.5)
    parser.add_argument('--ladder', action='store_true',
                       help='Enable the E1 escalation ladder above the recalibrator.')
    parser.add_argument('--ladder-target-coverage', type=float, default=0.9)
    parser.add_argument('--ladder-escalation-band', type=float, default=0.05)
    parser.add_argument('--ladder-escalation-persistence', type=int, default=5)
    parser.add_argument('--ladder-recovery-persistence', type=int, default=10)
    parser.add_argument('--ladder-widening-factor', type=float, default=1.5)
    parser.add_argument('--ladder-conservative-factor', type=float, default=3.0)
    args = parser.parse_args()

    # Backward-compat: --online-recal → --recalibrator rolling-origin.
    # If both are set explicitly to compatible values, --recalibrator wins.
    if args.online_recal and args.recalibrator == 'none':
        args.recalibrator = 'rolling-origin'
        logger.info("Backward-compat: --online-recal mapped to --recalibrator rolling-origin")
    elif args.online_recal and args.recalibrator != 'rolling-origin':
        logger.warning("--online-recal ignored: --recalibrator=%s takes precedence",
                       args.recalibrator)
    # Keep args.online_recal mirrored so the existing rolling-origin code path
    # uses the right flag regardless of which surface the user invoked.
    args.online_recal = (args.recalibrator == 'rolling-origin')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['start_time'] = datetime.now(timezone.utc).isoformat()
    with open(output_dir / 'controller_config.yaml', 'w') as f:
        yaml.dump(config, f)

    logger.info("Starting %s controller", args.method.upper())
    logger.info("  Model: %s", args.model_dir)
    logger.info("  Deployment: %s/%s", args.namespace, args.deployment)
    logger.info("  Duration: %ds, Interval: %ds", args.duration, args.interval)
    logger.info("  Hysteresis: %d, Replicas: %d-%d", args.hysteresis,
                 args.min_replicas, args.max_replicas)

    try:
        start_http_server(args.metrics_port)
        logger.info("Prometheus metrics endpoint: :%d/metrics", args.metrics_port)
    except Exception as e:
        logger.warning("Failed to start metrics server: %s (continuing without)", e)

    try:
        if args.method == 'be':
            uq = BootstrapEnsemble.load(args.model_dir, device='cpu')
        elif args.method == 'scp':
            uq = SplitConformal.load(args.model_dir, device='cpu')
        elif args.method == 'qr':
            qr = QuantileRegressor.load(args.model_dir, device='cpu')
            uq = qr
        logger.info("Loaded %s UQ model (h=%d, k=%d, alpha=%.2f)",
                     args.method.upper(), uq.h, uq.k, uq.alpha)
    except Exception as e:
        logger.error("Failed to load UQ model: %s", e)
        sys.exit(1)

    if args.online_recal and args.method != 'scp':
        logger.warning("--online-recal set with method=%s; only SCP is supported. "
                       "Disabling online recalibration.", args.method)
        args.online_recal = False
        if args.recalibrator == 'rolling-origin':
            args.recalibrator = 'none'

    # Only SCP exposes q_hat as a mutable calibrated quantile; BE/QR derive each interval.
    if args.recalibrator in ('aci', 'pid') and args.method != 'scp':
        logger.warning("--recalibrator=%s set with method=%s; only SCP is supported. "
                       "Disabling.", args.recalibrator, args.method)
        args.recalibrator = 'none'

    if args.ladder and not args.coverage_monitor:
        logger.info("--ladder requires --coverage-monitor; enabling implicitly "
                    "(target=%.3f, window=%d).",
                    args.coverage_target, args.coverage_window)
        args.coverage_monitor = True

    if args.recalibrator in ('aci', 'pid') and not args.coverage_monitor:
        logger.info("--recalibrator=%s benefits from --coverage-monitor; "
                    "enabling implicitly (target=%.3f, window=%d).",
                    args.recalibrator, args.coverage_target, args.coverage_window)
        args.coverage_monitor = True

    if args.lambda_risk is not None:
        logger.info("λ-risk mode active: λ=%.2f (tier policy formula overridden)",
                     args.lambda_risk)
    elif args.policy == 'ci-upper':
        logger.info("Policy=ci-upper: scaling to max(ci_upper); no tier policy.")

    current_replicas = args.min_replicas
    scale_log = []
    hysteresis_mgr = HysteresisManager()

    coverage_monitor: Optional[CoverageMonitor] = None
    if args.coverage_monitor:
        coverage_monitor = CoverageMonitor(
            window_size=args.coverage_window,
            target_coverage=args.coverage_target,
        )
        logger.info(
            "CoverageMonitor: window=%d, target=%.3f (Analyze only — no Plan action)",
            args.coverage_window, args.coverage_target,
        )

    # Each horizon needs recal_warmup matched observations before refreshing q_hat.
    recal_buffer: list[list[float]] = [[] for _ in range(uq.k)] if args.online_recal else []
    recal_updates = 0
    initial_q_hat = (
        uq.q_hat.copy()
        if args.recalibrator != 'none' and hasattr(uq, 'q_hat') and uq.q_hat is not None
        else None
    )

    # Seed each horizon so its conformal order statistic matches the offline q_hat;
    # this enables recalibration immediately without changing the initial interval.
    warmstart_n = 0
    if args.online_recal and args.recal_warmstart and initial_q_hat is not None:
        warmstart_n = args.recal_warmstart_n or args.recal_warmup
        rng = np.random.RandomState(args.recal_warmstart_seed)
        for h in range(uq.k):
            shape = np.abs(rng.randn(warmstart_n).astype(np.float64))  # half-normal, scale 1
            n_s = len(shape)
            q_index = min(int(np.ceil((1 - uq.alpha) * (n_s + 1))) - 1, n_s - 1)
            ref = float(np.sort(shape)[max(q_index, 0)])
            scale = float(uq.q_hat[h]) / ref if ref > 1e-9 else 0.0
            recal_buffer[h] = list(shape * scale)
        logger.info(
            "--recal-warmstart: seeded %d horizons x %d residuals; buffer (1-a) order "
            "statistic pinned to offline q_hat %s (online recal active from decision 0)",
            uq.k, warmstart_n, np.array2string(np.asarray(uq.q_hat), precision=3))

    # ACI/PID update only h0, using normalized residuals; prediction converts widths back to RPS.
    recalibrator: Optional[ConformalPID] = None
    if args.recalibrator == 'pid':
        recalibrator = ConformalPID(
            target_alpha=float(uq.alpha),
            k_p=args.pid_k_p,
            k_i=args.pid_k_i,
            k_d=args.pid_k_d,
            residual_buffer_size=args.recal_buffer_size,
            alpha_clip=(args.recal_alpha_clip_low, args.recal_alpha_clip_high),
        )
        logger.info(
            "ConformalPID active: target_α=%.3f, K_P=%.3f, K_I=%.4f, K_D=%.3f, "
            "buffer=%d, α_clip=(%.4g,%.3f)",
            recalibrator.target_alpha, recalibrator.k_p, recalibrator.k_i,
            recalibrator.k_d, args.recal_buffer_size,
            args.recal_alpha_clip_low, args.recal_alpha_clip_high,
        )
    elif args.recalibrator == 'aci':
        recalibrator = ACI(
            target_alpha=float(uq.alpha),
            eta=args.aci_eta,
            residual_buffer_size=args.recal_buffer_size,
            alpha_clip=(args.recal_alpha_clip_low, args.recal_alpha_clip_high),
        )
        logger.info(
            "ACI active: target_α=%.3f, η=%.3f, buffer=%d, α_clip=(%.4g,%.3f)",
            recalibrator.target_alpha, recalibrator.eta, args.recal_buffer_size,
            args.recal_alpha_clip_low, args.recal_alpha_clip_high,
        )

    ladder: Optional[EscalationLadder] = None
    if args.ladder:
        ladder = EscalationLadder(
            target_coverage=args.ladder_target_coverage,
            escalation_band=args.ladder_escalation_band,
            escalation_persistence=args.ladder_escalation_persistence,
            recovery_persistence=args.ladder_recovery_persistence,
            widening_factor=args.ladder_widening_factor,
            conservative_factor=args.ladder_conservative_factor,
        )
        logger.info(
            "EscalationLadder: target=%.3f, band=%.3f, escalate_K=%d, recover_K=%d, "
            "widening=%.2f, conservative=%.2f",
            args.ladder_target_coverage, args.ladder_escalation_band,
            args.ladder_escalation_persistence, args.ladder_recovery_persistence,
            args.ladder_widening_factor, args.ladder_conservative_factor,
        )

    kubectl_scale(args.context, args.namespace, args.deployment, current_replicas)
    time.sleep(5)

    shutdown = False

    def handle_signal(sig, frame):
        nonlocal shutdown
        logger.info("Received signal %s, shutting down...", sig)
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    start_time = time.time()
    end_time = start_time + args.duration

    logger.info("Control loop starting. Will run for %ds.", args.duration)

    while time.time() < end_time and not shutdown:
        loop_start = time.time()

        rps = query_prometheus(args.prometheus_url, args.rps_query)

        # Observation t scores forecast t-h-1 at horizon h.
        recal_updated_this_iter = False
        if args.online_recal and rps is not None and uq.norm_params is not None:
            sigma = max(float(uq.norm_params.sigma), 1e-6)
            for h in range(uq.k):
                idx = -(h + 1)
                if abs(idx) > len(scale_log):
                    continue
                old_entry = scale_log[idx]
                old_forecast = old_entry.get('point_forecast') or []
                if len(old_forecast) <= h:
                    continue
                # q_hat lives in normalized space; convert residual likewise.
                residual_norm = abs(float(rps) - float(old_forecast[h])) / sigma
                recal_buffer[h].append(residual_norm)
                if len(recal_buffer[h]) > args.recal_buffer_size:
                    recal_buffer[h] = recal_buffer[h][-args.recal_buffer_size:]

            if all(len(buf) >= args.recal_warmup for buf in recal_buffer):
                new_q_hat = np.zeros(uq.k, dtype=np.float32)
                for h in range(uq.k):
                    scores = np.sort(np.asarray(recal_buffer[h]))
                    n = len(scores)
                    q_index = min(int(np.ceil((1 - uq.alpha) * (n + 1))) - 1, n - 1)
                    new_q_hat[h] = scores[max(q_index, 0)]
                uq.q_hat = new_q_hat
                recal_updates += 1
                recal_updated_this_iter = True

        # Score the previous h0 interval; retain the offline quantile until a residual is available.
        recalibrator_updated_this_iter = False
        if (recalibrator is not None and rps is not None
                and uq.norm_params is not None and scale_log
                and hasattr(uq, 'q_hat') and uq.q_hat is not None):
            prior = scale_log[-1]
            prior_forecast = prior.get('point_forecast') or []
            prior_ci_lo = prior.get('ci_lower') or []
            prior_ci_hi = prior.get('ci_upper') or []
            if prior_forecast and prior_ci_lo and prior_ci_hi:
                sigma = max(float(uq.norm_params.sigma), 1e-6)
                residual_norm = abs(float(rps) - float(prior_forecast[0])) / sigma
                miscovered = not (
                    float(prior_ci_lo[0]) <= float(rps) <= float(prior_ci_hi[0])
                )
                recalibrator.update(residual_norm, miscovered)
                recalibrator_updated_this_iter = True
                try:
                    uq.q_hat[0] = float(recalibrator.quantile())
                except EmptyResidualBufferError:
                    pass  # warmup; keep prior q_hat[0]

        # Use prior coverage to widen h0 before issuing this interval.
        ladder_level = 0
        if ladder is not None and coverage_monitor is not None:
            ladder_level = ladder.step(coverage_monitor.trailing_coverage)
            if ladder_level > 0 and hasattr(uq, 'q_hat') and uq.q_hat is not None:
                uq.q_hat[0] = float(ladder.apply(float(uq.q_hat[0])))

        history = query_prometheus_range(
            args.prometheus_url, args.rps_query,
            lookback_s=args.history_length * 30
        )

        # Pad short history with the latest observation so cold starts can produce forecasts.
        seeded = False
        if 0 < len(history) < uq.h:
            pad_value = history[-1]
            history = [pad_value] * (uq.h - len(history)) + history
            seeded = True

        # 3. Predict with uncertainty
        decision_start = time.time()
        prediction_made = False
        pred = None

        if len(history) >= uq.h:
            try:
                history_array = np.array(history[-uq.h:], dtype=np.float32)
                pred = uq.predict_with_uncertainty(history_array)
                prediction_made = True
            except Exception as e:
                logger.warning("Prediction failed: %s", e)

        # Validate the previous interval before recording the next one.
        coverage_state = None
        if coverage_monitor is not None:
            if rps is not None:
                coverage_monitor.validate_pending(float(rps))
            if prediction_made and pred is not None:
                ci_lower_arr = pred.get('ci_lower')
                ci_upper_arr = pred.get('ci_upper')
                if (ci_lower_arr is not None and ci_upper_arr is not None
                        and len(ci_lower_arr) > 0 and len(ci_upper_arr) > 0):
                    coverage_monitor.record_prediction(
                        float(ci_lower_arr[0]),
                        float(ci_upper_arr[0]),
                    )
            g_trailing_coverage.set(float(coverage_monitor.trailing_coverage))
            g_coverage_shortfall.set(float(coverage_monitor.coverage_shortfall()))
            g_coverage_alert.set(int(coverage_monitor.alert_state))
            coverage_state = coverage_monitor.get_state()

        recalibrator_state = None
        if recalibrator is not None:
            recalibrator_state = recalibrator.state()
            g_conformal_pid_alpha.set(float(recalibrator_state['alpha']))
            g_conformal_pid_integral.set(float(recalibrator_state['integral']))
            g_conformal_pid_derivative.set(float(recalibrator_state['derivative']))
            if hasattr(uq, 'q_hat') and uq.q_hat is not None and len(uq.q_hat) > 0:
                g_conformal_pid_quantile_halfwidth.set(float(uq.q_hat[0]))
        ladder_state = None
        if ladder is not None:
            ladder_state = ladder.state()
            g_escalation_ladder_level.set(int(ladder_state['level']))

        if prediction_made and pred:
            raw_tier, raw_target = compute_target_replicas(
                point_forecast=pred['point_forecast'],
                confidence_score=pred['confidence_score'],
                ci_upper=pred.get('ci_upper'),
                slo_capacity=args.slo_capacity,
                target_util=args.target_util,
                safety_factor=args.safety_factor,
                min_replicas=args.min_replicas,
                max_replicas=args.max_replicas,
                policy=args.policy,
                lambda_risk=args.lambda_risk,
            )
        else:
            # No prediction available — conservative fallback
            raw_target = current_replicas
            raw_tier = 0

        tier, target = hysteresis_mgr.apply(
            proposed_tier=raw_tier,
            proposed_replicas=raw_target,
            current_replicas=current_replicas,
        )
        decision_latency = time.time() - decision_start

        # 6. Execute scaling (with min-delta threshold)
        delta = abs(target - current_replicas)
        if delta >= args.hysteresis:
            if target != current_replicas:
                old_replicas = current_replicas
                kubectl_scale(args.context, args.namespace, args.deployment, target)
                current_replicas = target
        else:
            target = current_replicas  # No-op: below hysteresis threshold

        observed_replicas = kubectl_get_replicas(args.context, args.namespace, args.deployment)

        try:
            _update_prometheus_metrics(
                prediction_made=prediction_made,
                pred=pred,
                tier=tier,
                target_replicas=target,
                current_replicas=observed_replicas if observed_replicas is not None else float("nan"),
                latency_s=decision_latency,
            )
        except Exception as e:
            logger.debug("Metrics update failed: %s", e)

        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'elapsed_s': round(time.time() - start_time, 1),
            'rps': rps,
            'point_forecast': pred['point_forecast'].tolist() if prediction_made and pred else [],
            'ci_lower': pred['ci_lower'].tolist() if prediction_made and pred else [],
            'ci_upper': pred['ci_upper'].tolist() if prediction_made and pred else [],
            'confidence_score': pred['confidence_score'] if prediction_made and pred else 0.0,
            'tier': tier,
            'raw_tier': raw_tier,
            'target_replicas': target,
            'raw_target': raw_target,
            'last_requested_replicas': current_replicas,
            'observed_replicas': observed_replicas,
            'prediction_made': prediction_made,
            'history_len': len(history),
            'seeded': seeded,
            'decision_latency_s': round(decision_latency, 4),
            'hysteresis': {
                'tier_duration': hysteresis_mgr.tier_duration,
                'downgrade_count': hysteresis_mgr.proposed_downgrade_count,
            },
        }
        if args.online_recal:
            log_entry['recal'] = {
                'updated': recal_updated_this_iter,
                'buffer_sizes': [len(buf) for buf in recal_buffer],
                'q_hat_used': uq.q_hat.tolist() if hasattr(uq, 'q_hat') and uq.q_hat is not None else None,
            }
        if recalibrator is not None and recalibrator_state is not None:
            log_entry['recalibrator'] = {
                'kind': args.recalibrator,
                'updated': recalibrator_updated_this_iter,
                'state': recalibrator_state,
            }
        if ladder is not None and ladder_state is not None:
            log_entry['ladder'] = ladder_state
        if coverage_state is not None:
            log_entry['coverage'] = coverage_state
        scale_log.append(log_entry)

        if prediction_made and pred:
            logger.info(
                "t=%4ds | rps=%.1f | forecast=%.1f | ci=[%.1f,%.1f] | "
                "conf=%.3f | raw_tier=%d → tier=%d | raw=%d → target=%d | cur=%d | "
                "lat=%.1fms",
                log_entry['elapsed_s'], rps or 0,
                pred['point_forecast'][0], pred['ci_lower'][0], pred['ci_upper'][0],
                pred['confidence_score'], raw_tier, tier,
                raw_target, target, current_replicas,
                decision_latency * 1000,
            )
        else:
            logger.info("t=%4ds | no prediction yet (history=%d)",
                         log_entry['elapsed_s'], len(history))

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
        logger.info("Scale log saved: %s (%d entries)", log_path, len(scale_log))
    except Exception as e:
        logger.error("Failed to save scale log to %s: %s", log_path, e)

    if scale_log:
        tiers_seen = set(e['tier'] for e in scale_log if e.get('tier', 0) > 0)
        total_scale_ops = sum(
            1 for i in range(1, len(scale_log))
            if scale_log[i]['target_replicas'] != scale_log[i-1]['target_replicas']
        )
        metrics_summary = {
            'total_intervals': len(scale_log),
            'total_scale_operations': total_scale_ops,
            'tiers_visited': sorted(tiers_seen),
            'final_replicas': current_replicas,
            'mean_decision_latency_ms': round(
                np.mean([e['decision_latency_s'] for e in scale_log]) * 1000, 2
            ) if scale_log else 0,
            'policy': args.policy,
            'lambda_risk': args.lambda_risk,
        }
        if args.online_recal:
            metrics_summary['online_recal'] = {
                'enabled': True,
                'q_hat_updates': recal_updates,
                'initial_q_hat': initial_q_hat.tolist() if initial_q_hat is not None else None,
                'final_q_hat': uq.q_hat.tolist() if hasattr(uq, 'q_hat') and uq.q_hat is not None else None,
                'buffer_size': args.recal_buffer_size,
                'warmup': args.recal_warmup,
                'warmstart': bool(args.recal_warmstart),
                'warmstart_n': warmstart_n,
                'final_buffer_sizes': [len(buf) for buf in recal_buffer],
            }
        if coverage_monitor is not None:
            total_validated = coverage_monitor.lifetime_validated
            total_covered = coverage_monitor.lifetime_covered
            coverage_rate = (
                total_covered / total_validated if total_validated > 0 else None
            )
            metrics_summary['coverage_monitor'] = {
                'enabled': True,
                'final_trailing_coverage': round(float(coverage_monitor.trailing_coverage), 4),
                'final_alert_state': int(coverage_monitor.alert_state),
                'total_validated': total_validated,
                'total_covered': total_covered,
                'coverage_rate': (round(coverage_rate, 4)
                                  if coverage_rate is not None else None),
                'target_coverage': float(coverage_monitor.target_coverage),
                'window_size': int(coverage_monitor.window_size),
            }
        if recalibrator is not None:
            metrics_summary['recalibrator'] = {
                'kind': args.recalibrator,
                'final_state': recalibrator.state(),
                'initial_q_hat': (
                    initial_q_hat.tolist() if initial_q_hat is not None else None
                ),
                'final_q_hat': (
                    uq.q_hat.tolist()
                    if hasattr(uq, 'q_hat') and uq.q_hat is not None else None
                ),
            }
        if ladder is not None:
            metrics_summary['ladder'] = ladder.state()
        with open(output_dir / 'operator_metrics_summary.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info("Metrics summary: %s", json.dumps(metrics_summary))

    logger.info("Controller finished. Final replicas: %d, tier: %d",
                 current_replicas, hysteresis_mgr.current_tier)


if __name__ == '__main__':
    main()
