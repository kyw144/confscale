#!/usr/bin/env python3
"""
Method Registry — scaling method configurations for the experiment orchestrator.

Each method implements:
    configure(namespace) -> bool
    reset(namespace) -> bool  
    get_state() -> dict

Registry for the methods retained in the paper artifact. Live execution
is unverified in this standalone layout; see docs/MAC_VERIFICATION.md.
"""

import logging
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

KUBE_CONTEXT = "kind-p3-experiments"
NAMESPACE = "infosys-benchmark"

# Thread-local kube-context override. Parallel run_matrix workers call
# set_thread_kube_context(...) at start; every kubectl()/controller-spawn
# call in this thread then targets that context instead of the global default.
# Single-threaded callers see KUBE_CONTEXT, preserving backward compat.
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


# ── Kubernetes Helpers ──────────────────────────────────────────────────

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


# ── Method Config Base ──────────────────────────────────────────────────

@dataclass
class MethodConfig:
    """A scaling method that can be configured and reset."""
    name: str
    description: str = ""
    # Per-cell run directory, set by the orchestrator before configure().
    # Controllers write controller_config.yaml + controller_scale_log.json here.
    # Each parallel worker has its own method instance (get_method returns
    # fresh copies), so this is thread-safe by construction.
    run_dir: Optional[Path] = None
    # Per-cell workload duration in seconds, set by the orchestrator before
    # configure(). Controllers compute their --duration ceiling from this so
    # the lifetime stays tied to the cell length rather than a magic number.
    cell_duration_s: Optional[int] = None

    # Grace period added on top of cell_duration_s for the controller's
    # --duration ceiling. Covers workload generator teardown timeout (120s),
    # rollout settling, reset() latency, and unexpected slop. 20 min is
    # deliberately generous — the controller is killed by reset() in normal
    # operation, this value only matters if reset() doesn't fire.
    CONTROLLER_DURATION_GRACE_S: int = 1200

    def configure(self, namespace: str = NAMESPACE) -> bool:
        raise NotImplementedError

    def reset(self, namespace: str = NAMESPACE) -> bool:
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        """Return current configuration for run_config.yaml snapshot."""
        raise NotImplementedError

    def controller_output_dir(self) -> Path:
        """Where the controller subprocess should write its logs.

        Prefers the per-cell run_dir; falls back to a global directory so
        ad-hoc invocations (smoke tests, manual runs) still work.
        """
        if self.run_dir is not None:
            return self.run_dir
        return Path(__file__).resolve().parent.parent / 'outputs' / 'controller'

    def controller_duration(self) -> int:
        """Lifetime ceiling (seconds) for the spawned controller subprocess.

        Derived from the cell duration plus a grace period; falls back to a
        2-hour ceiling for ad-hoc invocations where cell_duration_s isn't set.
        """
        if self.cell_duration_s is not None:
            return self.cell_duration_s + self.CONTROLLER_DURATION_GRACE_S
        return 7200

    @staticmethod
    def _clean_scaling_state(namespace: str = NAMESPACE) -> None:
        """Remove leftover autoscaling state from prior cell runs.

        Each parallel worker reuses the same kind cluster across multiple
        methods, so configure() must defensively clean up state that any
        prior method may have left behind:
          - hpa/compute-worker             (created by HPAMethod)
          - scaledobject/compute-worker-keda  (created by KEDAMethod)
          - hpa/keda-hpa-compute-worker-keda  (created by KEDA operator
            in response to the ScaledObject; survives if reset() races)
        ScaledObject is deleted first so KEDA's operator doesn't reconcile
        the HPA back into existence between the two delete calls. All
        deletes use --ignore-not-found so the helper is idempotent and
        safe to call from configure() even on the first run.
        """
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


# ── HPA-Reactive (Kubernetes HPA) ───────────────────────────────────────

@dataclass
class HPAMethod(MethodConfig):
    """Kubernetes HPA on CPU utilization.

    Two configure() paths, chosen by ``downscale_stabilization_s``:
      - ``None`` (default): the original v1 path — ``kubectl autoscale`` creates
        an autoscaling/v1 HPA whose downscale stabilization is the cluster-level
        kube-controller-manager default (300 s on these kind clusters). This is
        the path every pre-existing run used; left byte-for-byte intact.
      - set to an int: the v2 path (E-V7) — apply an autoscaling/v2 HPA with an
        explicit ``behavior`` block so the per-HPA scale-down stabilization
        window is tunable (the dominant cost lever on Pattern D, whose 300 s
        pulse period coincides with the v1 default window and pins the HPA at
        max). ``upscale_stabilization_s`` (default 0) reacts immediately up.

    Both paths share name/min/max/cpu_target and the same HPA object name, so
    ``_clean_scaling_state`` and ``reset`` are unchanged. v2 at
    ``downscale_stabilization_s=300`` reproduces the v1 default behaviour and
    serves as the E-V7 reproduction anchor.
    """

    name: str = "hpa-reactive"
    description: str = ("Kubernetes HPA on CPU utilization; v1 kubectl-autoscale "
                        "(cluster-default downscale) or v2 behavior block when "
                        "downscale_stabilization_s is set")
    min_replicas: int = 1
    max_replicas: int = 20
    cpu_target: int = 50
    # E-V7: when set (seconds), use the autoscaling/v2 behavior path with this
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

        # Ensure deployment exists at min_replicas
        if not kubectl_check([
            "scale", "deployment", "compute-worker",
            f"--replicas={self.min_replicas}", "-n", namespace,
        ]):
            return False

        # Wait for rollout
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

        # Verify HPA is active
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
        """Build the autoscaling/v2 HPA manifest (with behavior block) as YAML.

        Pure/stateless apart from reading self fields — unit-testable without a
        cluster. The scaleDown/scaleUp policies (Percent 100 / 15 s) match the
        Kubernetes v1 default policy, so v2 at stabilizationWindowSeconds=300
        reproduces the v1 default downscale behaviour exactly.
        """
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
        """E-V7 path: apply an autoscaling/v2 HPA with an explicit behavior block."""
        logger.info(
            "Configuring HPA-reactive (v2 behavior): min=%d max=%d cpu=%d%% "
            "downscale_stab=%ds upscale_stab=%ds",
            self.min_replicas, self.max_replicas, self.cpu_target,
            self.downscale_stabilization_s, self.upscale_stabilization_s,
        )

        # Same defensive cleanup as the v1 path (same HPA object name).
        self._clean_scaling_state(namespace)

        # Ensure deployment exists at min_replicas, then wait for rollout.
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

        # Apply the v2 HPA. Write the manifest to the per-cell run_dir for
        # provenance (falls back to a temp file for ad-hoc invocations).
        manifest = self._render_v2_hpa_manifest(namespace)
        if self.run_dir is not None:
            manifest_path = self.run_dir / "hpa_v2.yaml"
        else:
            manifest_path = Path(tempfile.gettempdir()) / f"hpa_v2_{namespace}.yaml"
        manifest_path.write_text(manifest)

        if not kubectl_check(["apply", "-f", str(manifest_path)]):
            return False

        # Verify HPA is active.
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
        # Scale back to 1 (HPA will take over from there)
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


# ── Static Replicas (no autoscaling — for profiling) ────────────────────

@dataclass
class StaticReplicasMethod(MethodConfig):
    """Fixed replica count, no autoscaling. Used for profiling and baselines."""

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


# ── Stub Methods (for future tasks) ─────────────────────────────────────

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


# ── Confidence-Aware Scaling Methods ──────────────────────────────────

# Paths for UQ models (trained by uq/train_all.py)
UQ_MODELS_DIR = Path(__file__).resolve().parent.parent / 'models' / 'uq'

# Controller script
CONTROLLER_SCRIPT = Path(__file__).resolve().parent / 'controller.py'
PYTHON = sys.executable  # artifact: use the active interpreter

# Workload pattern letter → model directory name
WORKLOAD_PATTERN_TO_MODEL = {
    'A': 'diurnal',
    'B': 'bursty',
    'C': 'batch_ramp',
    'D': 'signaling',
    # Also accept the full names directly
    'diurnal': 'diurnal',
    'bursty': 'bursty',
    'batch_ramp': 'batch_ramp',
    'signaling': 'signaling',
}

def _resolve_model_pattern(workload_pattern: str) -> str:
    """Map workload letter/full-name to model directory name. Defaults to diurnal."""
    return WORKLOAD_PATTERN_TO_MODEL.get(workload_pattern, 'diurnal')


@dataclass
class ConfScaleMethod(MethodConfig):
    """Confidence-aware scaling using UQ predictions.

    Starts a background controller process that polls Prometheus for RPS,
    runs UQ predictions, and scales the deployment according to the
    tiered confidence-aware policy.

    The choice of UQ method (BE/SCP/QR) is determined by which model
    directory is loaded.
    """

    uq_method: str = 'scp'  # 'be', 'scp', 'qr' — SCP is the only method hitting 88-91% on all 4 patterns
    workload_pattern: str = 'diurnal'  # A→diurnal, B→bursty, C→batch_ramp, D→signaling
    slo_capacity: float = 10.0
    target_util: float = 0.7
    safety_factor: float = 2.0
    min_replicas: int = 1
    max_replicas: int = 20
    prometheus_port: int = 9090
    # Scaling-decision knobs (forwarded to controller.py).
    # policy='tier' (default 3-tier), 'ci-upper' (MagicScaler-style baseline).
    # lambda_risk in [0,1] overrides policy when set: blends point and ci_upper.
    # online_recal: SCP-only; incrementally recalibrate q_hat from streaming residuals.
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
    # E1 recalibrator selection. 'none' = static SCP; 'rolling-origin' = the
    # historical online_recal path (kept for backward compat — registry's
    # confscale-rolling-origin alias maps here); 'aci' / 'pid' = the E1
    # recalibrators wired to the coverage monitor. SCP-only.
    recalibrator: str = 'none'
    pid_k_p: float = 0.1
    pid_k_i: float = 0.01
    pid_k_d: float = 0.05
    aci_eta: float = 0.1
    recal_alpha_clip_low: float = 1e-4
    recal_alpha_clip_high: float = 0.5
    # E1 coverage-conditional escalation ladder. Requires a coverage monitor,
    # which the controller auto-enables when this is True.
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
        # Resolve workload pattern to model directory name
        pattern_name = _resolve_model_pattern(self.workload_pattern)
        logger.info("Configuring %s (UQ=%s, pattern=%s → %s)...",
                     self.name, self.uq_method, self.workload_pattern, pattern_name)

        self._clean_scaling_state(namespace)

        # Set initial state
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
            logger.warning("UQ model not found at %s — falling back to HPA", model_dir)
            return HPAMethod(
                min_replicas=self.min_replicas,
                max_replicas=self.max_replicas,
            ).configure(namespace)

        # Start controller
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
        # E1 recalibrator forwarding. 'rolling-origin' is set via the same flag
        # but the controller honors --online-recal for backward compat — sending
        # both here would conflict if mismatched, so default-skip and trust the
        # online_recal flag for the rolling-origin path. ACI/PID get the new
        # flag and their gains.
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
        self.controller_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for controller to initialize
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

        # Restore baseline
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


# ── Baseline Methods ─────────────────────────────────────────────────

# Paths for baseline controller
BASELINE_CONTROLLER_SCRIPT = Path(__file__).resolve().parent.parent / 'baselines' / 'controller.py'
GRU_MODELS_DIR = Path(__file__).resolve().parent.parent / 'models' / 'gru'


@dataclass
class PredictiveMethod(MethodConfig):
    """HPA-Predictive baseline: GRU point forecast → direct replica override.

    Unlike HPA-reactive (which uses CPU threshold), this method bypasses HPA
    entirely and sets replica count based on predicted RPS.
    """

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

        # Set initial state
        if not kubectl_check([
            "scale", "deployment", "compute-worker",
            f"--replicas={self.min_replicas}", "-n", namespace,
        ]):
            return False

        kubectl([
            "rollout", "status", "deployment", "compute-worker",
            "-n", namespace, "--timeout=120s",
        ], timeout=130)

        # Resolve GRU model dir
        model_dir = GRU_MODELS_DIR / f"gru_compute-worker_{pattern_name}"
        if not model_dir.exists():
            logger.warning("GRU model not found at %s — falling back to HPA", model_dir)
            return HPAMethod(
                min_replicas=self.min_replicas,
                max_replicas=self.max_replicas,
            ).configure(namespace)

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
        self.controller_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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
            logger.warning("GRU model not found at %s — falling back to HPA", model_dir)
            return HPAMethod(
                min_replicas=self.min_replicas,
                max_replicas=self.max_replicas,
            ).configure(namespace)

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
        self.controller_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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
    """BASE-Inspired baseline: binary burst detection with two-mode policy.

    During bursts: conservative (1.5× forecast). Otherwise: precise (forecast directly).
    """

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
            logger.warning("GRU model not found at %s — falling back to HPA", model_dir)
            return HPAMethod(
                min_replicas=self.min_replicas,
                max_replicas=self.max_replicas,
            ).configure(namespace)

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
        self.controller_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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
    """Error-monitored baseline: trailing residual-MAE trigger.

    Tracks a sliding window of ``|point_forecast - observed_rps|``
    residuals. While the MAE stays below ``error_threshold`` the
    controller behaves like ``hpa-predictive``; once MAE crosses the
    threshold it escalates to ``forecast * error_safety_multiplier``.

    Designed as the C3 contrast for the Paper 3 reframe: it should
    catch level-drift workloads (pattern G) but miss volatility-drift
    workloads (pattern F) — the calibration-aware (coverage-monitored)
    controller is meant to win on the latter.
    """

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
            logger.warning("GRU model not found at %s — falling back to HPA", model_dir)
            return HPAMethod(
                min_replicas=self.min_replicas,
                max_replicas=self.max_replicas,
            ).configure(namespace)

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
        self.controller_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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
    """HPA-style controller fed by a UQ-emitted prediction interval.

    Architectural contrast to ConfScaleMethod: same UQ family is available,
    but the controller lives on the baseline path (single decision rule,
    no 3-tier ConfScale escalation, no recalibration). The CoverageMonitor
    is on by default — measuring the empirical coverage of the
    uncalibrated UQ interval is the entire point of this contrast.

    Decision rule (when ``decision_rule='upper_quantile'``):
        target_replicas = ceil(ci_upper_h0 / (slo_capacity * target_util))

    Only ``uq_method='qr'`` is supported today; the dataclass field is
    extensible for SCP/BE variants if needed later.
    """

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
            logger.warning("UQ model not found at %s — falling back to HPA", model_dir)
            return HPAMethod(
                min_replicas=self.min_replicas,
                max_replicas=self.max_replicas,
            ).configure(namespace)

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
        self.controller_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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
    """KEDA event-driven autoscaling baseline.

    Deploys a ScaledObject that triggers scaling on CPU > 70% via Prometheus.
    Represents the event-driven autoscaling paradigm (CNCF graduated).
    """

    name: str = "keda"
    description: str = "KEDA event-driven autoscaling (Prometheus scaler, CPU > 70%)"
    min_replicas: int = 1
    max_replicas: int = 20

    # Path to ScaledObject YAML
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

        # Apply ScaledObject
        if not kubectl_check([
            "apply", "-f", str(self._scaledobject_yaml),
            "-n", namespace,
        ]):
            logger.error("Failed to apply KEDA ScaledObject")
            return False

        time.sleep(5)

        # Verify ScaledObject is active
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

        # Delete ScaledObject
        kubectl_check([
            "delete", "scaledobject", "compute-worker-keda",
            "-n", namespace, "--ignore-not-found=true",
        ])

        # Scale back to 1
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


# ── Method Registry ─────────────────────────────────────────────────────

METHOD_REGISTRY: dict[str, MethodConfig] = {
    "hpa-reactive": HPAMethod(),
    "static": StaticReplicasMethod(),
    "static-2": StaticReplicasMethod(name="static-2", replicas=2),
    "static-4": StaticReplicasMethod(name="static-4", replicas=4),
    "static-8": StaticReplicasMethod(name="static-8", replicas=8),
    # UQ-driven confidence-aware methods
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
    # Rolling-origin SCP (Halkiewicz-style principled per-horizon recal).
    # Renamed from "confscale-scp-online" per the E1 brief — kept as a baseline
    # in the four-recalibrator experimental axis. Backward-compat alias is
    # added below the registry so existing data/p3_runs/ outputs still load.
    "confscale-rolling-origin": ConfScaleMethod(
        name="confscale-rolling-origin", uq_method="scp",
        description="SCP with rolling-origin per-horizon q_hat recalibration (Halkiewicz 2026)",
        online_recal=True,
        recalibrator="rolling-origin",
    ),
    # T7b: rolling-origin SCP WARM-STARTED — identical to confscale-rolling-origin
    # except the per-horizon residual buffers are pre-seeded from the offline q_hat
    # so online recal begins at decision 0 (no cold-start warmup under-coverage; see
    # T3/P3-D015). Additive variant; the un-warmstarted entry is unchanged.
    "confscale-rolling-origin-warmstart": ConfScaleMethod(
        name="confscale-rolling-origin-warmstart", uq_method="scp",
        description="SCP rolling-origin recal, warm-started buffers (T7b; q_hat-seeded at t=0)",
        online_recal=True,
        recalibrator="rolling-origin",
        recal_warmstart=True,
    ),
    # E1 ACI baseline (Gibbs & Candès NeurIPS 2021). Proportional-only
    # recalibrator wired to the h=0 coverage signal.
    "confscale-aci": ConfScaleMethod(
        name="confscale-aci", uq_method="scp",
        description="SCP + ACI (Gibbs-Candès) proportional recalibrator on h=0 coverage",
        recalibrator="aci",
        coverage_monitor=True,
    ),
    # E1 HEADLINE method (Angelopoulos, Candès, Tibshirani NeurIPS 2023).
    # Conformal PID recalibrator on h=0 coverage signal.
    "confscale-pid": ConfScaleMethod(
        name="confscale-pid", uq_method="scp",
        description="SCP + Conformal PID recalibrator (Angelopoulos et al. NeurIPS 2023)",
        recalibrator="pid",
        coverage_monitor=True,
    ),
    # E1 laddered variant — PID inner loop plus the coverage-conditional
    # escalation ladder as outer safety net.
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
    # Non-HPA baselines (Objective 06)
    "hpa-predictive": PredictiveMethod(),
    "hpa-predictive-safety": PredictiveSafetyMethod(),
    "keda": KEDAMethod(),
    "base-inspired": BASEInspiredMethod(),
    # Paper 3 reframe C3 contrast — point-error-monitored baseline.
    "hpa-error-monitored": ErrorMonitoredMethod(),
    # Paper 3 §6.8 H-divergence anchor (reframed) — HPA-style controller fed
    # by the QR upper quantile, no recalibration, coverage monitor enabled.
    # Architecturally clean coverage-comparable contrast to confscale-pid.
    "hpa-qr-monitored": HPAUQMethod(
        name="hpa-qr-monitored",
        description=("HPA-style controller with QR-emitted 90% prediction "
                     "intervals, no recalibration, coverage monitor enabled"),
        uq_method="qr",
        coverage_monitor=True,
    ),
}

# Backward-compat alias: the old "confscale-scp-online" name maps to the
# renamed "confscale-rolling-origin" entry so existing run outputs and
# scripts that reference the old key continue to work. New code should
# use the new name.
METHOD_REGISTRY["confscale-scp-online"] = METHOD_REGISTRY["confscale-rolling-origin"]


def get_method(name: str) -> MethodConfig:
    """Look up a method by name. Returns a fresh instance.

    Raises KeyError if not found.
    """
    if name not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method '{name}'. Available: {list(METHOD_REGISTRY.keys())}")

    base = METHOD_REGISTRY[name]
    # Return fresh instance for stateful methods
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
        # HPAMethod (and Static/Stub below) previously fell through to the
        # singleton `return base`, contradicting the "fresh instance" contract.
        # Benign while every hpa-reactive cell shared one config, but E-V7 runs
        # multiple HPA configs (cpu_target / downscale window) concurrently and
        # mutates them via _resolve_method_spec setattr — a fresh copy is
        # required so parallel workers don't clobber each other (and so the
        # setattr never pollutes the registry singleton).
        return replace(base)
    # NOTE: StaticReplicasMethod / StubMethod still return the singleton; safe
    # only as long as they aren't run in parallel with per-cell field overrides.
    return base


def list_methods() -> list[str]:
    """List all registered method names."""
    return list(METHOD_REGISTRY.keys())
