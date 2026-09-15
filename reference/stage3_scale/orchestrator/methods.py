#!/usr/bin/env python3
"""Scaling methods and controller subprocess lifecycle."""

import logging
import os
import sys
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

KUBE_CONTEXT = os.environ.get("CONFSCALE_KUBE_CONTEXT", "kind-confscale-experiments")
NAMESPACE = "infosys-benchmark"

# Each parallel worker routes kubectl and controller processes to its own context.
_thread_state = threading.local()


def set_thread_kube_context(context: str) -> None:
    """Override the kubectl context for the calling thread only."""
    _thread_state.kube_context = context


def clear_thread_kube_context() -> None:
    if hasattr(_thread_state, "kube_context"):
        del _thread_state.kube_context


def current_kube_context() -> str:
    """Return this thread's kube context, or the module default."""
    return getattr(_thread_state, "kube_context", KUBE_CONTEXT)


def kubectl(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    cmd = ["kubectl", f"--context={current_kube_context()}"] + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def kubectl_check(args: list[str], timeout: int = 30) -> bool:
    """Run kubectl, return True on success, log errors."""
    result = kubectl(args, timeout)
    if result.returncode != 0:
        logger.error("kubectl failed: %s\n%s", " ".join(args), result.stderr.strip())
        return False
    return True


@dataclass
class MethodConfig:
    """A scaling method that can be configured and reset."""
    name: str
    description: str = ""
    run_dir: Optional[Path] = None
    cell_duration_s: Optional[int] = None

    # Grace bounds orphaned controllers; reset normally stops them first.
    CONTROLLER_DURATION_GRACE_S: int = 1200

    def configure(self, namespace: str = NAMESPACE) -> bool:
        raise NotImplementedError

    def reset(self, namespace: str = NAMESPACE) -> bool:
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def controller_output_dir(self) -> Path:
        """Where the controller subprocess should write its logs."""
        if self.run_dir is not None:
            return self.run_dir
        return Path(__file__).resolve().parent.parent / 'outputs' / 'controller'

    def controller_duration(self) -> int:
        """Lifetime ceiling (seconds) for the spawned controller subprocess."""
        if self.cell_duration_s is not None:
            return self.cell_duration_s + self.CONTROLLER_DURATION_GRACE_S
        return 7200

    @staticmethod
    def _clean_scaling_state(namespace: str = NAMESPACE) -> None:
        kubectl([
            "delete", "scaledobject", "compute-worker-keda",
            "-n", namespace, "--ignore-not-found=true",
        ])
        kubectl([
            "delete", "hpa", "keda-hpa-compute-worker-keda",
            "-n", namespace, "--ignore-not-found=true",
        ])
        kubectl([
            "delete", "hpa", "compute-worker",
            "-n", namespace, "--ignore-not-found=true",
        ])


@dataclass
class HPAMethod(MethodConfig):
    """Kubernetes HPA on CPU utilization."""

    name: str = "hpa-reactive"
    description: str = ("Kubernetes HPA on CPU utilization; v1 kubectl-autoscale "
                        "(cluster-default downscale) or v2 behavior block when "
                        "downscale_stabilization_s is set")
    min_replicas: int = 1
    max_replicas: int = 20
    cpu_target: int = 50
    # When set (seconds), use the autoscaling/v2 behavior path with this
    # scale-down stabilization window. None => original v1 kubectl-autoscale path.
    downscale_stabilization_s: Optional[int] = None
    upscale_stabilization_s: int = 0

    def configure(self, namespace: str = NAMESPACE) -> bool:
        if self.downscale_stabilization_s is not None:
            return self._configure_v2(namespace)
        logger.info("Configuring HPA-reactive (v1): min=%d max=%d cpu=%d%%",
                     self.min_replicas, self.max_replicas, self.cpu_target)

        # Defensive cleanup: remove any HPA/ScaledObject left over from prior
        # cells on the same worker cluster.
        self._clean_scaling_state(namespace)

        if not kubectl_check([
            "scale", "deployment", "compute-worker",
            f"--replicas={self.min_replicas}", "-n", namespace,
        ]):
            return False

        kubectl([
            "rollout", "status", "deployment", "compute-worker",
            "-n", namespace, "--timeout=120s",
        ], timeout=130)

        time.sleep(2)

        # Create HPA (use --cpu with percentage format, not deprecated --cpu-percent)
        if not kubectl_check([
            "autoscale", "deployment", "compute-worker",
            f"--min={self.min_replicas}", f"--max={self.max_replicas}",
            f"--cpu={self.cpu_target}%", "-n", namespace,
        ]):
            return False

        time.sleep(3)
        result = kubectl([
            "get", "hpa", "compute-worker", "-n", namespace,
            "-o", "jsonpath={.status.currentReplicas}",
        ])
        if result.returncode != 0 or not result.stdout.strip():
            logger.warning("HPA not reporting currentReplicas — may need warmup")
        else:
            logger.info("HPA active: currentReplicas=%s", result.stdout.strip())

        return True

    def _render_v2_hpa_manifest(self, namespace: str = NAMESPACE) -> str:
        manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": "compute-worker", "namespace": namespace},
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "compute-worker",
                },
                "minReplicas": int(self.min_replicas),
                "maxReplicas": int(self.max_replicas),
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": int(self.cpu_target),
                        },
                    },
                }],
                "behavior": {
                    "scaleDown": {
                        "stabilizationWindowSeconds": int(self.downscale_stabilization_s),
                        "policies": [{"type": "Percent", "value": 100, "periodSeconds": 15}],
                    },
                    "scaleUp": {
                        "stabilizationWindowSeconds": int(self.upscale_stabilization_s),
                        "policies": [{"type": "Percent", "value": 100, "periodSeconds": 15}],
                    },
                },
            },
        }
        return yaml.safe_dump(manifest, sort_keys=False)

    def _configure_v2(self, namespace: str = NAMESPACE) -> bool:
        logger.info(
            "Configuring HPA-reactive (v2 behavior): min=%d max=%d cpu=%d%% "
            "downscale_stab=%ds upscale_stab=%ds",
            self.min_replicas, self.max_replicas, self.cpu_target,
            self.downscale_stabilization_s, self.upscale_stabilization_s,
        )

        # Same defensive cleanup as the v1 path (same HPA object name).
        self._clean_scaling_state(namespace)

        if not kubectl_check([
            "scale", "deployment", "compute-worker",
            f"--replicas={self.min_replicas}", "-n", namespace,
        ]):
            return False
        kubectl([
            "rollout", "status", "deployment", "compute-worker",
            "-n", namespace, "--timeout=120s",
        ], timeout=130)
        time.sleep(2)

        manifest = self._render_v2_hpa_manifest(namespace)
        if self.run_dir is not None:
            manifest_path = self.run_dir / "hpa_v2.yaml"
        else:
            manifest_path = Path(tempfile.gettempdir()) / f"hpa_v2_{namespace}.yaml"
        manifest_path.write_text(manifest)

        if not kubectl_check(["apply", "-f", str(manifest_path)]):
            return False

        time.sleep(3)
        result = kubectl([
            "get", "hpa", "compute-worker", "-n", namespace,
            "-o", "jsonpath={.status.currentReplicas}",
        ])
        if result.returncode != 0 or not result.stdout.strip():
            logger.warning("HPA(v2) not reporting currentReplicas — may need warmup")
        else:
            logger.info("HPA(v2) active: currentReplicas=%s", result.stdout.strip())

        return True

    def reset(self, namespace: str = NAMESPACE) -> bool:
        """Restore to baseline: 1 replica, HPA still active."""
        logger.info("Resetting HPA-reactive to baseline")
        kubectl_check([
            "scale", "deployment", "compute-worker",
            "--replicas=1", "-n", namespace,
        ])
        time.sleep(3)
        return True

    def get_state(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "cpu_target_pct": self.cpu_target,
            "downscale_stabilization_s": self.downscale_stabilization_s,
            "upscale_stabilization_s": self.upscale_stabilization_s,
            "hpa_api_version": (
                "autoscaling/v2" if self.downscale_stabilization_s is not None
                else "autoscaling/v1"
            ),
        }


@dataclass
class StaticReplicasMethod(MethodConfig):
    """Fixed replica count, no autoscaling."""

    name: str = "static"
    description: str = "Fixed replica count, no autoscaling"
    replicas: int = 1

    def configure(self, namespace: str = NAMESPACE) -> bool:
        logger.info("Configuring static: %d replicas", self.replicas)

        self._clean_scaling_state(namespace)

        if not kubectl_check([
            "scale", "deployment", "compute-worker",
            f"--replicas={self.replicas}", "-n", namespace,
        ]):
            return False

        kubectl([
            "rollout", "status", "deployment", "compute-worker",
            "-n", namespace, "--timeout=120s",
        ], timeout=130)
        return True

    def reset(self, namespace: str = NAMESPACE) -> bool:
        return kubectl_check([
            "scale", "deployment", "compute-worker",
            "--replicas=1", "-n", namespace,
        ])

    def get_state(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "replicas": self.replicas,
        }


@dataclass
class StubMethod(MethodConfig):
    """Placeholder for methods not yet implemented."""

    def configure(self, namespace: str = NAMESPACE) -> bool:
        logger.warning("Stub method '%s' — configure() not implemented", self.name)
        # Fall back to HPA-reactive so experiments can still run
        logger.info("Falling back to HPA-reactive for %s", self.name)
        return HPAMethod().configure(namespace)

    def reset(self, namespace: str = NAMESPACE) -> bool:
        return HPAMethod().reset(namespace)

    def get_state(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "status": "stub — not yet implemented",
        }


def _make_stub(name: str, description: str) -> StubMethod:
    return StubMethod(name=name, description=description)


MODELS_DIR = Path(os.environ.get('CONFSCALE_MODELS_DIR', Path(__file__).resolve().parent.parent / 'models'))
UQ_MODELS_DIR = MODELS_DIR / 'uq'

CONTROLLER_SCRIPT = Path(__file__).resolve().parent / 'controller.py'
PYTHON = sys.executable  # artifact: use the active interpreter

WORKLOAD_PATTERN_TO_MODEL = {
    'A': 'diurnal',
    'B': 'bursty',
    'C': 'batch_ramp',
    'D': 'signaling',
    'diurnal': 'diurnal',
    'bursty': 'bursty',
    'batch_ramp': 'batch_ramp',
    'signaling': 'signaling',
}

def _resolve_model_pattern(workload_pattern: str) -> str:
    return WORKLOAD_PATTERN_TO_MODEL.get(workload_pattern, 'diurnal')


@dataclass
class ConfScaleMethod(MethodConfig):
    """Confidence-aware scaling using UQ predictions."""

    uq_method: str = 'scp'  # 'be', 'scp', 'qr' — SCP is the only method hitting 88-91% on all 4 patterns
    workload_pattern: str = 'diurnal'  # A→diurnal, B→bursty, C→batch_ramp, D→signaling
    slo_capacity: float = 10.0
    target_util: float = 0.7
    safety_factor: float = 2.0
    min_replicas: int = 1
    max_replicas: int = 20
    prometheus_port: int = 9090
    policy: str = "tier"
    lambda_risk: Optional[float] = None
    online_recal: bool = False
    recal_buffer_size: int = 200
    recal_warmup: int = 30
    # T7b: warm-start the rolling-origin recalibrator (pre-seed buffers from the
    # offline q_hat so online recal begins at decision 0). Additive, default off.
    recal_warmstart: bool = False
    recal_warmstart_n: Optional[int] = None
    recal_warmstart_seed: int = 0
    recalibrator: str = 'none'
    pid_k_p: float = 0.1
    pid_k_i: float = 0.01
    pid_k_d: float = 0.05
    aci_eta: float = 0.1
    recal_alpha_clip_low: float = 1e-4
    recal_alpha_clip_high: float = 0.5
    ladder: bool = False
    ladder_target_coverage: float = 0.9
    ladder_escalation_band: float = 0.05
    ladder_escalation_persistence: int = 5
    ladder_recovery_persistence: int = 10
    ladder_widening_factor: float = 1.5
    ladder_conservative_factor: float = 3.0
    # CoverageMonitor knobs (also forwarded to controller.py). The monitor is
    # auto-enabled by the controller when recalibrator in {aci, pid} or ladder.
    coverage_monitor: bool = False
    coverage_target: float = 0.9
    coverage_window: int = 30
    controller_process: Optional[subprocess.Popen] = None

    def configure(self, namespace: str = NAMESPACE) -> bool:
        import atexit
        pattern_name = _resolve_model_pattern(self.workload_pattern)
        logger.info("Configuring %s (UQ=%s, pattern=%s → %s)...",
                     self.name, self.uq_method, self.workload_pattern, pattern_name)

        self._clean_scaling_state(namespace)

        if not kubectl_check([
            "scale", "deployment", "compute-worker",
            f"--replicas={self.min_replicas}", "-n", namespace,
        ]):
            return False

        kubectl([
            "rollout", "status", "deployment", "compute-worker",
            "-n", namespace, "--timeout=120s",
        ], timeout=130)

        # Determine model directory — auto-select based on workload pattern
        model_dir = UQ_MODELS_DIR / pattern_name / self.uq_method
        if not model_dir.exists():
            raise FileNotFoundError(f"Required model missing: {model_dir}; refusing HPA fallback")

        output_dir = self.controller_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(PYTHON), str(CONTROLLER_SCRIPT),
            '--method', self.uq_method,
            '--model-dir', str(model_dir),
            '--namespace', namespace,
            '--prometheus-url', f'http://localhost:{self.prometheus_port}',
            '--slo-capacity', str(self.slo_capacity),
            '--target-util', str(self.target_util),
            '--safety-factor', str(self.safety_factor),
            '--min-replicas', str(self.min_replicas),
            '--max-replicas', str(self.max_replicas),
            '--interval', '30',
            '--duration', str(self.controller_duration()),  # ceiling — reset() normally kills it first
            '--output-dir', str(output_dir),
            '--context', current_kube_context(),
            '--policy', self.policy,
        ]
        if self.lambda_risk is not None:
            cmd += ['--lambda-risk', str(self.lambda_risk)]
        if self.online_recal:
            cmd += [
                '--online-recal',
                '--recal-buffer-size', str(self.recal_buffer_size),
                '--recal-warmup', str(self.recal_warmup),
            ]
            if self.recal_warmstart:
                cmd += ['--recal-warmstart',
                        '--recal-warmstart-seed', str(self.recal_warmstart_seed)]
                if self.recal_warmstart_n is not None:
                    cmd += ['--recal-warmstart-n', str(self.recal_warmstart_n)]
        # Rolling-origin uses --online-recal; avoid conflicting recalibrator flags.
        if self.recalibrator in ('aci', 'pid'):
            cmd += [
                '--recalibrator', self.recalibrator,
                '--recal-buffer-size', str(self.recal_buffer_size),
                '--recal-alpha-clip-low', str(self.recal_alpha_clip_low),
                '--recal-alpha-clip-high', str(self.recal_alpha_clip_high),
            ]
            if self.recalibrator == 'pid':
                cmd += [
                    '--pid-k-p', str(self.pid_k_p),
                    '--pid-k-i', str(self.pid_k_i),
                    '--pid-k-d', str(self.pid_k_d),
                ]
            else:
                cmd += ['--aci-eta', str(self.aci_eta)]
        if self.coverage_monitor:
            cmd += [
                '--coverage-monitor',
                '--coverage-target', str(self.coverage_target),
                '--coverage-window', str(self.coverage_window),
            ]
        if self.ladder:
            cmd += [
                '--ladder',
                '--ladder-target-coverage', str(self.ladder_target_coverage),
                '--ladder-escalation-band', str(self.ladder_escalation_band),
                '--ladder-escalation-persistence', str(self.ladder_escalation_persistence),
                '--ladder-recovery-persistence', str(self.ladder_recovery_persistence),
                '--ladder-widening-factor', str(self.ladder_widening_factor),
                '--ladder-conservative-factor', str(self.ladder_conservative_factor),
            ]

        logger.info("Starting controller: %s", ' '.join(cmd[:4]) + ' ...')
        with open(output_dir / "controller.log", "a") as controller_log:
            self.controller_process = subprocess.Popen(
                cmd, stdout=controller_log, stderr=subprocess.STDOUT, text=True,
            )

        time.sleep(5)
        if self.controller_process.poll() is not None:
            logger.error("Controller exited immediately (rc=%d)", self.controller_process.returncode)
            return False

        logger.info("Controller started (pid=%d)", self.controller_process.pid)
        return True

    def reset(self, namespace: str = NAMESPACE) -> bool:
        logger.info("Stopping %s controller...", self.name)
        if self.controller_process:
            self.controller_process.terminate()
            try:
                self.controller_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.controller_process.kill()
                self.controller_process.wait()
            logger.info("Controller stopped")

        kubectl_check([
            "scale", "deployment", "compute-worker",
            "--replicas=1", "-n", namespace,
        ])
        return True

    def get_state(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "uq_method": self.uq_method,
            "workload_pattern": self.workload_pattern,
            "slo_capacity": self.slo_capacity,
            "target_util": self.target_util,
            "safety_factor": self.safety_factor,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "policy": self.policy,
            "lambda_risk": self.lambda_risk,
            "online_recal": self.online_recal,
            "recal_buffer_size": self.recal_buffer_size,
            "recal_warmup": self.recal_warmup,
            "recal_warmstart": self.recal_warmstart,
            "recal_warmstart_n": self.recal_warmstart_n,
            "recal_warmstart_seed": self.recal_warmstart_seed,
            "recalibrator": self.recalibrator,
            "pid_k_p": self.pid_k_p,
            "pid_k_i": self.pid_k_i,
            "pid_k_d": self.pid_k_d,
            "aci_eta": self.aci_eta,
            "recal_alpha_clip_low": self.recal_alpha_clip_low,
            "recal_alpha_clip_high": self.recal_alpha_clip_high,
            "ladder": self.ladder,
            "ladder_target_coverage": self.ladder_target_coverage,
            "ladder_escalation_band": self.ladder_escalation_band,
            "ladder_escalation_persistence": self.ladder_escalation_persistence,
            "ladder_recovery_persistence": self.ladder_recovery_persistence,
            "ladder_widening_factor": self.ladder_widening_factor,
            "ladder_conservative_factor": self.ladder_conservative_factor,
            "coverage_monitor": self.coverage_monitor,
            "coverage_target": self.coverage_target,
            "coverage_window": self.coverage_window,
        }


BASELINE_CONTROLLER_SCRIPT = Path(__file__).resolve().parent.parent / 'baselines' / 'controller.py'
GRU_MODELS_DIR = MODELS_DIR / 'gru'


@dataclass
class PredictiveMethod(MethodConfig):
    """HPA-Predictive baseline: GRU point forecast → direct replica override."""

    name: str = "hpa-predictive"
    description: str = "GRU point forecast → ceil(forecast / capacity), no HPA"
    slo_capacity: float = 10.0
    target_util: float = 0.7
    min_replicas: int = 1
    max_replicas: int = 20
    prometheus_port: int = 9090
    workload_pattern: str = 'diurnal'  # A→diurnal, B→bursty, C→batch_ramp, D→signaling
    controller_process: Optional[subprocess.Popen] = None

    def configure(self, namespace: str = NAMESPACE) -> bool:
        pattern_name = _resolve_model_pattern(self.workload_pattern)
        logger.info("Configuring %s (GRU: %s → %s)...", self.name, self.workload_pattern, pattern_name)

        self._clean_scaling_state(namespace)

        if not kubectl_check([
            "scale", "deployment", "compute-worker",
            f"--replicas={self.min_replicas}", "-n", namespace,
        ]):
            return False

        kubectl([
            "rollout", "status", "deployment", "compute-worker",
            "-n", namespace, "--timeout=120s",
        ], timeout=130)

        model_dir = GRU_MODELS_DIR / f"gru_compute-worker_{pattern_name}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Required model missing: {model_dir}; refusing HPA fallback")

        output_dir = self.controller_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(PYTHON), str(BASELINE_CONTROLLER_SCRIPT),
            '--mode', 'predictive',
            '--model-dir', str(model_dir),
            '--namespace', namespace,
            '--prometheus-url', f'http://localhost:{self.prometheus_port}',
            '--slo-capacity', str(self.slo_capacity),
            '--target-util', str(self.target_util),
            '--min-replicas', str(self.min_replicas),
            '--max-replicas', str(self.max_replicas),
            '--interval', '30',
            '--duration', str(self.controller_duration()),
            '--output-dir', str(output_dir),
            '--context', current_kube_context(),
        ]

        logger.info("Starting baseline controller: %s ...", ' '.join(cmd[:4]))
        with open(output_dir / "controller.log", "a") as controller_log:
            self.controller_process = subprocess.Popen(
                cmd, stdout=controller_log, stderr=subprocess.STDOUT, text=True,
            )

        time.sleep(5)
        if self.controller_process.poll() is not None:
            logger.error("Baseline controller exited immediately (rc=%d)",
                         self.controller_process.returncode)
            return False

        logger.info("Baseline controller started (pid=%d)", self.controller_process.pid)
        return True

    def reset(self, namespace: str = NAMESPACE) -> bool:
        logger.info("Stopping %s controller...", self.name)
        if self.controller_process:
            self.controller_process.terminate()
            try:
                self.controller_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.controller_process.kill()
                self.controller_process.wait()
            logger.info("Controller stopped")

        kubectl_check([
            "scale", "deployment", "compute-worker",
            "--replicas=1", "-n", namespace,
        ])
        return True

    def get_state(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "workload_pattern": self.workload_pattern,
            "slo_capacity": self.slo_capacity,
            "target_util": self.target_util,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
        }


@dataclass
class PredictiveSafetyMethod(PredictiveMethod):
    """HPA-Predictive-Safety: point forecast + 20% safety margin."""

    name: str = "hpa-predictive-safety"
    description: str = "GRU forecast + 20%% safety margin → direct replica override"
    safety_margin: float = 0.2

    def configure(self, namespace: str = NAMESPACE) -> bool:
        logger.info("Configuring %s (margin: %.0f%%)...",
                     self.name, self.safety_margin * 100)

        self._clean_scaling_state(namespace)

        if not kubectl_check([
            "scale", "deployment", "compute-worker",
            f"--replicas={self.min_replicas}", "-n", namespace,
        ]):
            return False

        kubectl([
            "rollout", "status", "deployment", "compute-worker",
            "-n", namespace, "--timeout=120s",
        ], timeout=130)

        pattern_name = _resolve_model_pattern(self.workload_pattern)
        model_dir = GRU_MODELS_DIR / f"gru_compute-worker_{pattern_name}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Required model missing: {model_dir}; refusing HPA fallback")

        output_dir = self.controller_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(PYTHON), str(BASELINE_CONTROLLER_SCRIPT),
            '--mode', 'predictive-safety',
            '--model-dir', str(model_dir),
            '--namespace', namespace,
            '--prometheus-url', f'http://localhost:{self.prometheus_port}',
            '--slo-capacity', str(self.slo_capacity),
            '--target-util', str(self.target_util),
            '--safety-margin', str(self.safety_margin),
            '--min-replicas', str(self.min_replicas),
            '--max-replicas', str(self.max_replicas),
            '--interval', '30',
            '--duration', str(self.controller_duration()),
            '--output-dir', str(output_dir),
            '--context', current_kube_context(),
        ]

        logger.info("Starting baseline controller: %s ...", ' '.join(cmd[:4]))
        with open(output_dir / "controller.log", "a") as controller_log:
            self.controller_process = subprocess.Popen(
                cmd, stdout=controller_log, stderr=subprocess.STDOUT, text=True,
            )

        time.sleep(5)
        if self.controller_process.poll() is not None:
            logger.error("Baseline controller exited immediately (rc=%d)",
                         self.controller_process.returncode)
            return False

        logger.info("Baseline controller started (pid=%d)", self.controller_process.pid)
        return True

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["safety_margin"] = self.safety_margin
        return state


@dataclass
class BASEInspiredMethod(PredictiveMethod):
    """BASE-Inspired baseline: binary burst detection with two-mode policy."""

    name: str = "base-inspired"
    description: str = "Binary burst detection → conservative (1.5×) or precise policy"
    burst_multiplier: float = 1.5
    burst_threshold: float = 2.0

    def configure(self, namespace: str = NAMESPACE) -> bool:
        logger.info("Configuring %s (burst×%.1f, threshold=%.1f)...",
                     self.name, self.burst_multiplier, self.burst_threshold)

        self._clean_scaling_state(namespace)

        if not kubectl_check([
            "scale", "deployment", "compute-worker",
            f"--replicas={self.min_replicas}", "-n", namespace,
        ]):
            return False

        kubectl([
            "rollout", "status", "deployment", "compute-worker",
            "-n", namespace, "--timeout=120s",
        ], timeout=130)

        pattern_name = _resolve_model_pattern(self.workload_pattern)
        model_dir = GRU_MODELS_DIR / f"gru_compute-worker_{pattern_name}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Required model missing: {model_dir}; refusing HPA fallback")

        output_dir = self.controller_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(PYTHON), str(BASELINE_CONTROLLER_SCRIPT),
            '--mode', 'base-inspired',
            '--model-dir', str(model_dir),
            '--namespace', namespace,
            '--prometheus-url', f'http://localhost:{self.prometheus_port}',
            '--slo-capacity', str(self.slo_capacity),
            '--target-util', str(self.target_util),
            '--burst-multiplier', str(self.burst_multiplier),
            '--burst-threshold', str(self.burst_threshold),
            '--min-replicas', str(self.min_replicas),
            '--max-replicas', str(self.max_replicas),
            '--interval', '30',
            '--duration', str(self.controller_duration()),
            '--output-dir', str(output_dir),
            '--context', current_kube_context(),
        ]

        logger.info("Starting baseline controller: %s ...", ' '.join(cmd[:4]))
        with open(output_dir / "controller.log", "a") as controller_log:
            self.controller_process = subprocess.Popen(
                cmd, stdout=controller_log, stderr=subprocess.STDOUT, text=True,
            )

        time.sleep(5)
        if self.controller_process.poll() is not None:
            logger.error("Baseline controller exited immediately (rc=%d)",
                         self.controller_process.returncode)
            return False

        logger.info("Baseline controller started (pid=%d)", self.controller_process.pid)
        return True

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["burst_multiplier"] = self.burst_multiplier
        state["burst_threshold"] = self.burst_threshold
        return state


@dataclass
class ErrorMonitoredMethod(PredictiveMethod):
    """Error-monitored baseline: trailing residual-MAE trigger."""

    name: str = "hpa-error-monitored"
    description: str = (
        "GRU forecast + trailing residual-MAE trigger; conservative "
        "scale when MAE exceeds threshold"
    )
    error_window: int = 20
    error_threshold: float = 5.0
    error_safety_multiplier: float = 1.5

    def configure(self, namespace: str = NAMESPACE) -> bool:
        logger.info(
            "Configuring %s (window=%d, threshold=%.2f, safety=%.2f)...",
            self.name, self.error_window, self.error_threshold,
            self.error_safety_multiplier,
        )

        self._clean_scaling_state(namespace)

        if not kubectl_check([
            "scale", "deployment", "compute-worker",
            f"--replicas={self.min_replicas}", "-n", namespace,
        ]):
            return False

        kubectl([
            "rollout", "status", "deployment", "compute-worker",
            "-n", namespace, "--timeout=120s",
        ], timeout=130)

        pattern_name = _resolve_model_pattern(self.workload_pattern)
        model_dir = GRU_MODELS_DIR / f"gru_compute-worker_{pattern_name}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Required model missing: {model_dir}; refusing HPA fallback")

        output_dir = self.controller_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(PYTHON), str(BASELINE_CONTROLLER_SCRIPT),
            '--mode', 'error-monitored',
            '--model-dir', str(model_dir),
            '--namespace', namespace,
            '--prometheus-url', f'http://localhost:{self.prometheus_port}',
            '--slo-capacity', str(self.slo_capacity),
            '--target-util', str(self.target_util),
            '--error-window', str(self.error_window),
            '--error-threshold', str(self.error_threshold),
            '--error-safety-multiplier', str(self.error_safety_multiplier),
            '--min-replicas', str(self.min_replicas),
            '--max-replicas', str(self.max_replicas),
            '--interval', '30',
            '--duration', str(self.controller_duration()),
            '--output-dir', str(output_dir),
            '--context', current_kube_context(),
        ]

        logger.info("Starting baseline controller: %s ...", ' '.join(cmd[:4]))
        with open(output_dir / "controller.log", "a") as controller_log:
            self.controller_process = subprocess.Popen(
                cmd, stdout=controller_log, stderr=subprocess.STDOUT, text=True,
            )

        time.sleep(5)
        if self.controller_process.poll() is not None:
            logger.error("Baseline controller exited immediately (rc=%d)",
                         self.controller_process.returncode)
            return False

        logger.info("Baseline controller started (pid=%d)", self.controller_process.pid)
        return True

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["error_window"] = self.error_window
        state["error_threshold"] = self.error_threshold
        state["error_safety_multiplier"] = self.error_safety_multiplier
        return state


@dataclass
class HPAUQMethod(MethodConfig):
    """HPA-style controller fed by a UQ-emitted prediction interval."""

    uq_method: str = 'qr'
    coverage_monitor: bool = True
    coverage_target: float = 0.90
    coverage_window: int = 30
    decision_rule: str = 'upper_quantile'  # forward-compat; only value supported now
    slo_capacity: float = 10.0
    target_util: float = 0.7
    min_replicas: int = 1
    max_replicas: int = 20
    prometheus_port: int = 9090
    workload_pattern: str = 'diurnal'  # A→diurnal, B→bursty, C→batch_ramp, D→signaling
    controller_process: Optional[subprocess.Popen] = None

    def configure(self, namespace: str = NAMESPACE) -> bool:
        pattern_name = _resolve_model_pattern(self.workload_pattern)
        logger.info("Configuring %s (UQ=%s, pattern=%s → %s, rule=%s)...",
                    self.name, self.uq_method, self.workload_pattern,
                    pattern_name, self.decision_rule)

        self._clean_scaling_state(namespace)

        if not kubectl_check([
            "scale", "deployment", "compute-worker",
            f"--replicas={self.min_replicas}", "-n", namespace,
        ]):
            return False

        kubectl([
            "rollout", "status", "deployment", "compute-worker",
            "-n", namespace, "--timeout=120s",
        ], timeout=130)

        # Resolve UQ model dir — same convention as ConfScaleMethod.
        model_dir = UQ_MODELS_DIR / pattern_name / self.uq_method
        if not model_dir.exists():
            raise FileNotFoundError(f"Required model missing: {model_dir}; refusing HPA fallback")

        output_dir = self.controller_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(PYTHON), str(BASELINE_CONTROLLER_SCRIPT),
            '--mode', 'hpa-qr-monitored',
            '--model-dir', str(model_dir),
            '--namespace', namespace,
            '--prometheus-url', f'http://localhost:{self.prometheus_port}',
            '--slo-capacity', str(self.slo_capacity),
            '--target-util', str(self.target_util),
            '--min-replicas', str(self.min_replicas),
            '--max-replicas', str(self.max_replicas),
            '--interval', '30',
            '--duration', str(self.controller_duration()),
            '--output-dir', str(output_dir),
            '--context', current_kube_context(),
        ]
        if self.coverage_monitor:
            cmd += [
                '--coverage-monitor',
                '--coverage-target', str(self.coverage_target),
                '--coverage-window', str(self.coverage_window),
            ]

        logger.info("Starting baseline controller: %s ...", ' '.join(cmd[:4]))
        with open(output_dir / "controller.log", "a") as controller_log:
            self.controller_process = subprocess.Popen(
                cmd, stdout=controller_log, stderr=subprocess.STDOUT, text=True,
            )

        time.sleep(5)
        if self.controller_process.poll() is not None:
            logger.error("Baseline controller exited immediately (rc=%d)",
                         self.controller_process.returncode)
            return False

        logger.info("Baseline controller started (pid=%d)", self.controller_process.pid)
        return True

    def reset(self, namespace: str = NAMESPACE) -> bool:
        logger.info("Stopping %s controller...", self.name)
        if self.controller_process:
            self.controller_process.terminate()
            try:
                self.controller_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.controller_process.kill()
                self.controller_process.wait()
            logger.info("Controller stopped")

        kubectl_check([
            "scale", "deployment", "compute-worker",
            "--replicas=1", "-n", namespace,
        ])
        return True

    def get_state(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "uq_method": self.uq_method,
            "workload_pattern": self.workload_pattern,
            "slo_capacity": self.slo_capacity,
            "target_util": self.target_util,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "coverage_monitor": self.coverage_monitor,
            "coverage_target": self.coverage_target,
            "coverage_window": self.coverage_window,
            "decision_rule": self.decision_rule,
        }


@dataclass
class KEDAMethod(MethodConfig):
    """KEDA event-driven autoscaling baseline."""

    name: str = "keda"
    description: str = "KEDA event-driven autoscaling (Prometheus scaler, CPU > 70%)"
    min_replicas: int = 1
    max_replicas: int = 20

    _scaledobject_yaml = Path(__file__).resolve().parent.parent / 'baselines' / 'keda-scaledobject.yaml'

    def configure(self, namespace: str = NAMESPACE) -> bool:
        logger.info("Configuring KEDA (Prometheus scaler, CPU > 70%%)...")

        self._clean_scaling_state(namespace)

        # Scale to min before applying
        if not kubectl_check([
            "scale", "deployment", "compute-worker",
            f"--replicas={self.min_replicas}", "-n", namespace,
        ]):
            return False

        kubectl([
            "rollout", "status", "deployment", "compute-worker",
            "-n", namespace, "--timeout=120s",
        ], timeout=130)

        if not kubectl_check([
            "apply", "-f", str(self._scaledobject_yaml),
            "-n", namespace,
        ]):
            logger.error("Failed to apply KEDA ScaledObject")
            return False

        time.sleep(5)

        result = kubectl([
            "get", "scaledobject", "compute-worker-keda",
            "-n", namespace,
            "-o", "jsonpath={.status.conditions[?(@.type=='Ready')].status}",
        ])
        if result.returncode == 0 and 'True' in result.stdout:
            logger.info("KEDA ScaledObject Ready")
        else:
            logger.warning("KEDA ScaledObject may not be ready yet: %s",
                          result.stdout.strip())

        return True

    def reset(self, namespace: str = NAMESPACE) -> bool:
        logger.info("Removing KEDA ScaledObject...")

        # Delete HPA that KEDA created (KEDA manages an internal HPA)
        kubectl([
            "delete", "hpa", "keda-hpa-compute-worker-keda",
            "-n", namespace, "--ignore-not-found=true",
        ])

        kubectl_check([
            "delete", "scaledobject", "compute-worker-keda",
            "-n", namespace, "--ignore-not-found=true",
        ])

        kubectl_check([
            "scale", "deployment", "compute-worker",
            "--replicas=1", "-n", namespace,
        ])
        return True

    def get_state(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "scaler_type": "Prometheus",
            "cpu_threshold_pct": 70,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
        }


METHOD_REGISTRY: dict[str, MethodConfig] = {
    "hpa-reactive": HPAMethod(),
    "static": StaticReplicasMethod(),
    "static-2": StaticReplicasMethod(name="static-2", replicas=2),
    "static-4": StaticReplicasMethod(name="static-4", replicas=4),
    "static-8": StaticReplicasMethod(name="static-8", replicas=8),
    "confscale-be": ConfScaleMethod(
        name="confscale-be", uq_method="be",
        description="Confidence-aware scaling with bootstrap ensemble UQ",
    ),
    "confscale-qr": ConfScaleMethod(
        name="confscale-qr", uq_method="qr",
        description="Confidence-aware scaling with quantile regression UQ",
    ),
    "confscale-scp": ConfScaleMethod(
        name="confscale-scp", uq_method="scp",
        description="Confidence-aware scaling with split-conformal UQ",
    ),
    # MagicScaler-style risk-quantile baseline: scales to ci_upper directly,
    # no tier policy. Engages explicitly with nearest prior (Pan et al., PVLDB 2023).
    "risk-quantile-scp": ConfScaleMethod(
        name="risk-quantile-scp", uq_method="scp",
        description="MagicScaler-style: scale to SCP ci_upper, no tier policy",
        policy="ci-upper",
    ),
    "confscale-rolling-origin": ConfScaleMethod(
        name="confscale-rolling-origin", uq_method="scp",
        description="SCP with rolling-origin per-horizon q_hat recalibration (Halkiewicz 2026)",
        online_recal=True,
        recalibrator="rolling-origin",
    ),
    # Seed buffers from offline quantiles so recalibration starts at the first decision.
    "confscale-rolling-origin-warmstart": ConfScaleMethod(
        name="confscale-rolling-origin-warmstart", uq_method="scp",
        description="SCP rolling-origin recal, warm-started buffers (T7b; q_hat-seeded at t=0)",
        online_recal=True,
        recalibrator="rolling-origin",
        recal_warmstart=True,
    ),
    "confscale-aci": ConfScaleMethod(
        name="confscale-aci", uq_method="scp",
        description="SCP + ACI (Gibbs-Candès) proportional recalibrator on h=0 coverage",
        recalibrator="aci",
        coverage_monitor=True,
    ),
    "confscale-pid": ConfScaleMethod(
        name="confscale-pid", uq_method="scp",
        description="SCP + Conformal PID recalibrator (Angelopoulos et al. NeurIPS 2023)",
        recalibrator="pid",
        coverage_monitor=True,
    ),
    "confscale-pid-laddered": ConfScaleMethod(
        name="confscale-pid-laddered", uq_method="scp",
        description="SCP + Conformal PID + coverage-conditional escalation ladder",
        recalibrator="pid",
        coverage_monitor=True,
        ladder=True,
    ),
    "confscale-aci-laddered": ConfScaleMethod(
        name="confscale-aci-laddered", uq_method="scp",
        description="SCP + ACI (Gibbs-Candès) + coverage-conditional escalation ladder",
        recalibrator="aci",
        coverage_monitor=True,
        ladder=True,
    ),
    "confscale-rolling-origin-laddered": ConfScaleMethod(
        name="confscale-rolling-origin-laddered", uq_method="scp",
        description="SCP + rolling-origin per-horizon recalibrator + coverage-conditional escalation ladder",
        online_recal=True,
        recalibrator="rolling-origin",
        coverage_monitor=True,
        ladder=True,
    ),
    "hpa-predictive": PredictiveMethod(),
    "hpa-predictive-safety": PredictiveSafetyMethod(),
    "keda": KEDAMethod(),
    "base-inspired": BASEInspiredMethod(),
    "hpa-error-monitored": ErrorMonitoredMethod(),
    "hpa-qr-monitored": HPAUQMethod(
        name="hpa-qr-monitored",
        description=("HPA-style controller with QR-emitted 90% prediction "
                     "intervals, no recalibration, coverage monitor enabled"),
        uq_method="qr",
        coverage_monitor=True,
    ),
}

# Keep the old method name readable in existing outputs and scripts.
METHOD_REGISTRY["confscale-scp-online"] = METHOD_REGISTRY["confscale-rolling-origin"]


def get_method(name: str) -> MethodConfig:
    """Look up a method by name."""
    if name not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method '{name}'. Available: {list(METHOD_REGISTRY.keys())}")

    base = METHOD_REGISTRY[name]
    if isinstance(base, ConfScaleMethod):
        return ConfScaleMethod(
            name=base.name,
            description=base.description,
            uq_method=base.uq_method,
            slo_capacity=base.slo_capacity,
            target_util=base.target_util,
            safety_factor=base.safety_factor,
            min_replicas=base.min_replicas,
            max_replicas=base.max_replicas,
            policy=base.policy,
            lambda_risk=base.lambda_risk,
            online_recal=base.online_recal,
            recal_buffer_size=base.recal_buffer_size,
            recal_warmup=base.recal_warmup,
            recal_warmstart=base.recal_warmstart,
            recal_warmstart_n=base.recal_warmstart_n,
            recal_warmstart_seed=base.recal_warmstart_seed,
            recalibrator=base.recalibrator,
            pid_k_p=base.pid_k_p,
            pid_k_i=base.pid_k_i,
            pid_k_d=base.pid_k_d,
            aci_eta=base.aci_eta,
            recal_alpha_clip_low=base.recal_alpha_clip_low,
            recal_alpha_clip_high=base.recal_alpha_clip_high,
            ladder=base.ladder,
            ladder_target_coverage=base.ladder_target_coverage,
            ladder_escalation_band=base.ladder_escalation_band,
            ladder_escalation_persistence=base.ladder_escalation_persistence,
            ladder_recovery_persistence=base.ladder_recovery_persistence,
            ladder_widening_factor=base.ladder_widening_factor,
            ladder_conservative_factor=base.ladder_conservative_factor,
            coverage_monitor=base.coverage_monitor,
            coverage_target=base.coverage_target,
            coverage_window=base.coverage_window,
        )
    if isinstance(base, PredictiveSafetyMethod):
        return PredictiveSafetyMethod(
            name=base.name,
            description=base.description,
            slo_capacity=base.slo_capacity,
            target_util=base.target_util,
            min_replicas=base.min_replicas,
            max_replicas=base.max_replicas,
            workload_pattern=base.workload_pattern,
            safety_margin=base.safety_margin,
        )
    if isinstance(base, BASEInspiredMethod):
        return BASEInspiredMethod(
            name=base.name,
            description=base.description,
            slo_capacity=base.slo_capacity,
            target_util=base.target_util,
            min_replicas=base.min_replicas,
            max_replicas=base.max_replicas,
            workload_pattern=base.workload_pattern,
            burst_multiplier=base.burst_multiplier,
            burst_threshold=base.burst_threshold,
        )
    if isinstance(base, ErrorMonitoredMethod):
        return ErrorMonitoredMethod(
            name=base.name,
            description=base.description,
            slo_capacity=base.slo_capacity,
            target_util=base.target_util,
            min_replicas=base.min_replicas,
            max_replicas=base.max_replicas,
            workload_pattern=base.workload_pattern,
            error_window=base.error_window,
            error_threshold=base.error_threshold,
            error_safety_multiplier=base.error_safety_multiplier,
        )
    if isinstance(base, PredictiveMethod):
        return PredictiveMethod(
            name=base.name,
            description=base.description,
            slo_capacity=base.slo_capacity,
            target_util=base.target_util,
            min_replicas=base.min_replicas,
            max_replicas=base.max_replicas,
            workload_pattern=base.workload_pattern,
        )
    if isinstance(base, HPAUQMethod):
        return HPAUQMethod(
            name=base.name,
            description=base.description,
            uq_method=base.uq_method,
            coverage_monitor=base.coverage_monitor,
            coverage_target=base.coverage_target,
            coverage_window=base.coverage_window,
            decision_rule=base.decision_rule,
            slo_capacity=base.slo_capacity,
            target_util=base.target_util,
            min_replicas=base.min_replicas,
            max_replicas=base.max_replicas,
            workload_pattern=base.workload_pattern,
        )
    if isinstance(base, HPAMethod):
        # Return a fresh method so cell overrides cannot mutate the registry.
        return replace(base)
    return replace(base)


def list_methods() -> list[str]:
    """List all registered method names."""
    return list(METHOD_REGISTRY.keys())
