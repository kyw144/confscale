#!/usr/bin/env python3
"""
Experiment Matrix Orchestrator — runs the full P3 experimental matrix.

For each (method × workload × replicate) cell:
    1. Configure the scaling method
    2. Run the workload generator
    3. Collect metrics from Prometheus
    4. Save run_config.yaml + outputs
    5. Reset to baseline

Usage:
    # Run full matrix from config
    python run_matrix.py --config matrix.yaml

    # Run a single cell for testing
    python run_matrix.py --method hpa-reactive --workload A --replicate 1 --duration 120

    # Dry-run: validate config without executing
    python run_matrix.py --config matrix.yaml --dry-run

Contract:
    run_matrix(config_path, output_dir) -> dict
"""

# Local artifact reference entrypoint; cluster behavior is unverified.
if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")


import argparse
import concurrent.futures
import csv
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

# Add parent to path so imports work in both package mode and script mode
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from collector.collect import collect_metrics

try:
    from .methods import (
        METHOD_REGISTRY, get_method, MethodConfig, KUBE_CONTEXT, NAMESPACE, kubectl,
        set_thread_kube_context, clear_thread_kube_context,
    )
    from .clusters import (
        WorkerSlot, make_slots,
        DEFAULT_CLUSTER_PREFIX, DEFAULT_BASE_FRONTEND_PORT, DEFAULT_BASE_PROMETHEUS_PORT,
    )
except ImportError:
    from orchestrator.methods import (
        METHOD_REGISTRY, get_method, MethodConfig, KUBE_CONTEXT, NAMESPACE, kubectl,
        set_thread_kube_context, clear_thread_kube_context,
    )
    from orchestrator.clusters import (
        WorkerSlot, make_slots,
        DEFAULT_CLUSTER_PREFIX, DEFAULT_BASE_FRONTEND_PORT, DEFAULT_BASE_PROMETHEUS_PORT,
    )

logger = logging.getLogger(__name__)


def _resolve_method_spec(spec) -> "MethodConfig":
    """Resolve a yaml method entry into a fresh MethodConfig instance.

    Forms accepted:
      - "confscale-scp"                                         # bare name
      - {name: confscale-scp, max_replicas: 30}                 # name + overrides
      - {base: confscale-scp, name: confscale-scp-lambda-0.0,   # rename + overrides
         lambda_risk: 0.0}                                       # for sweeps where
                                                                 # multiple cells share
                                                                 # a registry base.
    """
    if isinstance(spec, str):
        return get_method(spec)
    if isinstance(spec, dict):
        spec = dict(spec)
        base_name = spec.pop("base", spec.get("name"))
        if base_name is None:
            raise KeyError(f"method spec missing 'name' or 'base': {spec!r}")
        method = get_method(base_name)
        for k, v in spec.items():
            if hasattr(method, k):
                setattr(method, k, v)
        return method
    raise TypeError(f"unsupported method spec type: {type(spec).__name__}")


# ── Constants ───────────────────────────────────────────────────────────

WORKLOAD_GEN_SCRIPT = Path(__file__).resolve().parent.parent / "workload_gen.py"
PYTHON = Path(__file__).resolve().parent.parent.parent / ".venv" / "bin" / "python"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "matrix.yaml"

PROMETHEUS_NS = "monitoring"
PROMETHEUS_SVC = "prometheus-kube-prometheus-prometheus"
PROMETHEUS_PORT = 9090
FRONTEND_NODEPORT = 30080


# ── Port-Forward Manager ────────────────────────────────────────────────

class PortForward:
    """Manage a kubectl port-forward as a subprocess."""

    def __init__(self, namespace: str, service: str, local_port: int,
                 remote_port: int, kube_context: str = KUBE_CONTEXT):
        self.namespace = namespace
        self.service = service
        self.local_port = local_port
        self.remote_port = remote_port
        self.kube_context = kube_context
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        cmd = [
            "kubectl", f"--context={self.kube_context}",
            "port-forward", "-n", self.namespace,
            f"svc/{self.service}",
            f"{self.local_port}:{self.remote_port}",
        ]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        logger.info("Port-forward: %s:%d → %s/%s:%d",
                     self.service, self.local_port,
                     self.namespace, self.service, self.remote_port)

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("Port-forward stopped: %s:%d", self.service, self.local_port)


# ── Health Checks ────────────────────────────────────────────────────────

def check_cluster_ready() -> bool:
    """Verify the kind cluster and namespace are accessible."""
    result = kubectl(["get", "ns", NAMESPACE, "-o", "name"])
    if result.returncode != 0:
        logger.error("Cannot access namespace '%s'. Is the cluster running?", NAMESPACE)
        return False

    # Check compute-worker deployment exists
    result = kubectl(["get", "deployment", "compute-worker", "-n", NAMESPACE])
    if result.returncode != 0:
        logger.error("compute-worker deployment not found in %s", NAMESPACE)
        return False

    logger.info("Cluster check: OK (namespace=%s, deployment=compute-worker)", NAMESPACE)
    return True


def check_frontend_reachable(frontend_url: str, timeout: int = 5) -> bool:
    """Check that the frontend health endpoint responds."""
    import urllib.request
    try:
        url = f"{frontend_url.rstrip('/')}/health"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                logger.info("Frontend reachable: %s", url)
                return True
    except Exception as e:
        logger.error("Frontend not reachable at %s: %s", url, e)
    return False


def check_disk_space(output_dir: Path, min_free_gb: float = 5.0) -> bool:
    """Ensure at least min_free_gb GB free on the output filesystem."""
    try:
        stat = os.statvfs(output_dir)
        free_bytes = stat.f_frsize * stat.f_bavail
        free_gb = free_bytes / (1024 ** 3)
        if free_gb < min_free_gb:
            logger.error("Low disk space: %.1f GB free (need %.1f GB)", free_gb, min_free_gb)
            return False
        logger.info("Disk space: %.1f GB free (OK)", free_gb)
        return True
    except Exception as e:
        logger.warning("Could not check disk space: %s", e)
        return True  # Don't block on disk check failure


# ── Run Log CSV ─────────────────────────────────────────────────────────

class RunLog:
    """Append-only CSV log of all runs executed by this orchestrator session.

    Thread-safe: callers may invoke log() concurrently from multiple workers.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields = [
            "run_id", "worker_id", "method", "workload", "replicate", "status",
            "duration_s", "start_time", "end_time",
            "slo_violation_rate", "p95_ms", "mean_replicas",
            "total_requests", "error_rate", "error_message",
        ]
        self._lock = threading.Lock()
        exists = self.path.exists()
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fields)
        if not exists:
            self._writer.writeheader()
            self._file.flush()

    def log(self, **kwargs):
        row = {k: kwargs.get(k, "") for k in self._fields}
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()

    def close(self):
        with self._lock:
            self._file.close()


# ── Single Run Executor ──────────────────────────────────────────────────

def execute_single_run(
    method: MethodConfig,
    workload_pattern: str,
    replicate: int,
    duration_s: int,
    complexity: int,
    frontend_url: str,
    prometheus_url: str,
    output_dir: Path,
    workload_extra_args: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Execute ONE cell of the experiment matrix.

    Returns:
        dict with status, run_id, and summary metrics.
    """
    run_start = datetime.now(timezone.utc)
    run_id = f"{method.name}_{workload_pattern.lower()}_rep{replicate}_{run_start.strftime('%Y%m%d_%H%M%S')}"
    # E-V6 fix (Anomaly A4): resolve to an absolute path so the workload
    # generator subprocess (launched with cwd=run_dir) cannot re-resolve a
    # relative --output-dir against its own cwd and double the path
    # (<run_dir>/data/p3_runs/outputs/<batch>/<run_id>/workload_*.csv).
    # A relative --output-dir was the trigger; the doubled trace was then
    # invisible to the flat run_dir.glob() below, so collect.py never saw a
    # trace_csv_path and metrics.json got no e2e.* block.
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RUN: %s", run_id)
    logger.info("  Method:   %s", method.name)
    logger.info("  Workload: %s", workload_pattern)
    logger.info("  Replicate: %d", replicate)
    logger.info("  Duration:  %ds", duration_s)
    logger.info("=" * 60)

    status = "ok"
    error_message = ""

    try:
        # 0. Tell the method which workload pattern is about to run
        # (maps to correct model dir: A→diurnal, B→bursty, etc.)
        if hasattr(method, 'workload_pattern'):
            method.workload_pattern = workload_pattern

        # 0b. Per-cell controller output directory. Each cell has a fresh
        # method instance (get_method returns copies for stateful methods),
        # so this is thread-safe under parallel workers.
        method.run_dir = run_dir

        # 0c. Tell the method how long the workload will run, so the
        # controller's --duration ceiling tracks cell length + grace
        # rather than a hardcoded magic number.
        method.cell_duration_s = duration_s

        # 1. Configure method
        logger.info("Step 1/5: Configuring %s...", method.name)
        if not method.configure(NAMESPACE):
            raise RuntimeError(f"Method {method.name} configure() failed")
        time.sleep(5)  # Let HPA stabilize

        # 2. Record start time
        start_time = datetime.now(timezone.utc)
        logger.info("Step 2/5: Starting workload at %s", start_time.isoformat())

        # 3. Run workload generator
        logger.info("Step 3/5: Running workload pattern %s for %ds...", workload_pattern, duration_s)
        workload_cmd = [
            str(PYTHON), str(WORKLOAD_GEN_SCRIPT),
            workload_pattern,
            "--duration", str(duration_s),
            "--complexity", str(complexity),
            "--target", frontend_url,
            "--output-dir", str(run_dir),
        ]

        # Add pattern-specific args
        if workload_pattern == "B":
            workload_cmd.extend(["--rps-base", "30"])
        elif workload_pattern == "C":
            workload_cmd.extend(["--rps-base", "20", "--rps-peak", "150"])

        # Caller-supplied extra workload args (e.g. F/G drift-timing pilot-freeze:
        # --drift-start/--drift-window/--noise-start/--noise-end). Additive; when
        # absent the workload generator uses its own defaults (300/600/3/25).
        if workload_extra_args:
            workload_cmd.extend([str(a) for a in workload_extra_args])

        proc = subprocess.run(
            workload_cmd,
            capture_output=True,
            text=True,
            timeout=duration_s + 120,  # 2 min grace period
            cwd=str(run_dir),
        )

        if proc.returncode != 0:
            logger.warning("Workload generator exit code: %d", proc.returncode)
            stderr_tail = proc.stderr.strip()[-500:] if proc.stderr else ""
            if stderr_tail:
                logger.warning("Stderr: ...%s", stderr_tail)

        end_time = datetime.now(timezone.utc)
        actual_duration = int((end_time - start_time).total_seconds())
        logger.info("Workload complete: %ds actual", actual_duration)

        # 3b. Find workload trace CSV for E2E SLO metrics
        trace_csv_files = list(run_dir.glob("workload_*_timeseries.csv"))
        trace_csv_path = trace_csv_files[0] if trace_csv_files else None
        if trace_csv_path:
            logger.info("Found workload trace: %s", trace_csv_path)
        else:
            logger.warning("No workload trace CSV found in %s", run_dir)

        # 4. Collect metrics
        logger.info("Step 4/5: Collecting metrics from Prometheus...")
        try:
            summary = collect_metrics(
                prometheus_url=prometheus_url,
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                method=method.name,
                workload=workload_pattern,
                replicate=replicate,
                output_dir=run_dir,
                do_snapshot=False,  # Skip TSDB snapshots by default; enable for key runs
                trace_csv_path=trace_csv_path,
            )
            logger.info("  Status: %s, p95=%s, violations=%s",
                         summary.get("status"), summary.get("p95_ms"),
                         summary.get("slo_violation_rate"))
        except Exception as e:
            logger.error("Metrics collection failed: %s", e)
            summary = {
                "status": "collection_failed",
                "error": str(e),
                "method": method.name,
                "workload": workload_pattern,
                "replicate": replicate,
            }

        # 5. Write run_config.yaml
        logger.info("Step 5/5: Saving run configuration...")
        run_config = {
            "run_id": run_id,
            "method": method.name,
            "method_config": method.get_state(),
            "workload": workload_pattern,
            "replicate": replicate,
            "duration_s": duration_s,
            "actual_duration_s": actual_duration,
            "complexity": complexity,
            "frontend_url": frontend_url,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "status": summary.get("status", "unknown"),
        }
        config_path = run_dir / "run_config.yaml"
        config_path.write_text(yaml.dump(run_config, default_flow_style=False))
        logger.info("  Config saved: %s", config_path)

    except subprocess.TimeoutExpired:
        status = "timeout"
        error_message = f"Workload generator timed out after {duration_s + 120}s"
        end_time = datetime.now(timezone.utc)
        summary = {"status": "timeout", "error": error_message}
        logger.error(error_message)
    except Exception as e:
        status = "failed"
        error_message = str(e)
        end_time = datetime.now(timezone.utc)
        summary = {"status": "failed", "error": error_message}
        logger.error("Run failed: %s", e)
        traceback.print_exc()
    finally:
        # 6. Reset method
        try:
            logger.info("Resetting %s to baseline...", method.name)
            method.reset(NAMESPACE)
        except Exception as e:
            logger.warning("Reset failed (non-fatal): %s", e)

    result = {
        "run_id": run_id,
        "method": method.name,
        "workload": workload_pattern,
        "replicate": replicate,
        "status": status,
        "error_message": error_message,
        "duration_s": duration_s,
        "start_time": start_time.isoformat() if 'start_time' in dir() else run_start.isoformat(),
        "end_time": end_time.isoformat(),
        "summary": summary,
        "output_dir": str(run_dir),
    }
    logger.info("Run %s: %s", run_id, status)
    return result


# ── Matrix Runner ────────────────────────────────────────────────────────

def run_matrix(
    config_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
    workers: int = 1,
    cluster_prefix: str = DEFAULT_CLUSTER_PREFIX,
    base_frontend_port: int = DEFAULT_BASE_FRONTEND_PORT,
    base_prometheus_port: int = DEFAULT_BASE_PROMETHEUS_PORT,
    stagger_seconds: int = 30,
) -> dict[str, Any]:
    """
    Execute the full experimental matrix from a YAML config.

    Args:
        config_path: Path to matrix.yaml
        output_dir: Root output directory for per-run subdirectories
        dry_run: If True, print the matrix plan without executing
        workers: Parallel workers (default 1 = serial path, unchanged).
                 When > 1, expects N kind clusters from setup_parallel_clusters.py.
        cluster_prefix: Cluster name prefix.
        base_frontend_port: Worker i uses base_frontend_port + i.
        base_prometheus_port: Worker i uses base_prometheus_port + i.
        stagger_seconds: Per-worker startup delay to decorrelate workload spikes.

    Returns:
        dict with status, runs_completed, runs_failed, run_summaries
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    methods_names = config.get("methods", [])
    workloads = config.get("workloads", [])
    replicates = config.get("replicates", 3)
    # Allow runs to start replicate numbering above 1 (e.g. to tack reps 4-5
    # onto a prior 1-3 run without renaming cells post-hoc).
    start_replicate = config.get("start_replicate", 1)
    duration_s = config.get("duration_s", 3600)
    complexity = config.get("complexity", 100000)
    cooldown_s = config.get("cooldown_s", 60)
    frontend_url = config.get("frontend_url", f"http://localhost:{FRONTEND_NODEPORT}")

    # Resolve methods
    methods: list[MethodConfig] = []
    for m in methods_names:
        methods.append(_resolve_method_spec(m))

    total_cells = len(methods) * len(workloads) * replicates
    logger.info("Matrix: %d method(s) × %d workload(s) × %d replicate(s) = %d cells",
                 len(methods), len(workloads), replicates, total_cells)
    logger.info("Duration per cell: %ds, cooldown: %ds, workers: %d",
                 duration_s, cooldown_s, workers)
    serial_seconds = total_cells * (duration_s + cooldown_s)
    parallel_seconds = serial_seconds / max(workers, 1)
    logger.info("Estimated wall time: %.1f h serial, %.1f h with %d workers",
                 serial_seconds / 3600, parallel_seconds / 3600, workers)

    if dry_run:
        logger.info("DRY RUN — printing matrix plan:")
        for method in methods:
            for wl in workloads:
                for rep in range(start_replicate, start_replicate + replicates):
                    logger.info("  %s | %s | rep %d", method.name, wl, rep)
        if workers > 1:
            slots = make_slots(
                workers,
                cluster_prefix=cluster_prefix,
                base_frontend_port=base_frontend_port,
                base_prometheus_port=base_prometheus_port,
            )
            logger.info("Parallel slots:")
            for s in slots:
                logger.info(
                    "  %s context=%s frontend=%d prom=%d",
                    s.label(), s.kube_context, s.frontend_port, s.prometheus_port,
                )
        return {"status": "dry_run", "cells": total_cells, "runs": []}

    # Dispatch to parallel branch when workers > 1
    if workers > 1:
        return _run_matrix_parallel(
            methods_names=methods_names,
            workloads=workloads,
            replicates=replicates,
            start_replicate=start_replicate,
            duration_s=duration_s,
            complexity=complexity,
            cooldown_s=cooldown_s,
            output_dir=Path(output_dir),
            workers=workers,
            cluster_prefix=cluster_prefix,
            base_frontend_port=base_frontend_port,
            base_prometheus_port=base_prometheus_port,
            stagger_seconds=stagger_seconds,
        )

    # Health checks
    if not check_cluster_ready():
        return {"status": "cluster_not_ready", "runs_completed": 0, "runs_failed": 0, "runs": []}
    if not check_frontend_reachable(frontend_url):
        return {"status": "frontend_not_reachable", "runs_completed": 0, "runs_failed": 0, "runs": []}
    if not check_disk_space(output_dir):
        return {"status": "low_disk_space", "runs_completed": 0, "runs_failed": 0, "runs": []}

    # Start Prometheus port-forward
    prometheus_pf = PortForward(
        namespace=PROMETHEUS_NS,
        service=PROMETHEUS_SVC,
        local_port=PROMETHEUS_PORT,
        remote_port=PROMETHEUS_PORT,
    )
    prometheus_pf.start()
    prometheus_url = f"http://localhost:{PROMETHEUS_PORT}"

    # Run log
    output_dir = Path(output_dir)
    run_log = RunLog(output_dir / "run_log.csv")

    run_summaries = []
    runs_completed = 0
    runs_failed = 0
    terminated = False

    def handle_sigint(sig, frame):
        nonlocal terminated
        logger.warning("\nSIGINT received — finishing current run, then exiting...")
        terminated = True

    original_handler = signal.signal(signal.SIGINT, handle_sigint)

    try:
        for method in methods:
            for workload in workloads:
                for rep in range(start_replicate, start_replicate + replicates):
                    if terminated:
                        logger.info("Terminated by user. %d/%d cells completed.", runs_completed, total_cells)
                        break

                    result = execute_single_run(
                        method=method,
                        workload_pattern=workload,
                        replicate=rep,
                        duration_s=duration_s,
                        complexity=complexity,
                        frontend_url=frontend_url,
                        prometheus_url=prometheus_url,
                        output_dir=output_dir,
                    )

                    # Log to run_log.csv
                    summary = result.get("summary", {})
                    run_log.log(
                        run_id=result["run_id"],
                        method=result["method"],
                        workload=result["workload"],
                        replicate=result["replicate"],
                        status=result["status"],
                        duration_s=result.get("duration_s", duration_s),
                        start_time=result.get("start_time", ""),
                        end_time=result.get("end_time", ""),
                        slo_violation_rate=summary.get("slo_violation_rate", ""),
                        p95_ms=summary.get("p95_ms", ""),
                        mean_replicas=summary.get("mean_replicas", ""),
                        total_requests=summary.get("total_requests", ""),
                        error_rate=summary.get("error_rate", ""),
                        error_message=result.get("error_message", ""),
                    )

                    run_summaries.append(result)
                    if result["status"] == "ok":
                        runs_completed += 1
                    else:
                        runs_failed += 1

                    # Cooldown
                    if not terminated and (method != methods[-1] or workload != workloads[-1] or rep < replicates):
                        logger.info("Cooldown: %ds...", cooldown_s)
                        time.sleep(cooldown_s)

                if terminated:
                    break
            if terminated:
                break

    finally:
        signal.signal(signal.SIGINT, original_handler)
        prometheus_pf.stop()
        run_log.close()

    logger.info("=" * 60)
    logger.info("MATRIX COMPLETE: %d completed, %d failed, %d total cells",
                 runs_completed, runs_failed, runs_completed + runs_failed)
    logger.info("Run log: %s", output_dir / "run_log.csv")

    return {
        "status": "complete" if not terminated else "terminated",
        "runs_completed": runs_completed,
        "runs_failed": runs_failed,
        "total_cells": total_cells,
        "runs": run_summaries,
    }


# ── Parallel Matrix Runner ──────────────────────────────────────────────

def _apply_slot_to_method(method: MethodConfig, slot: WorkerSlot) -> None:
    """Stamp slot-specific values onto a method instance.

    ConfScale/Predictive/PredictiveSafety/BASEInspired all carry an instance
    `prometheus_port` field (default 9090). For parallel mode each worker
    talks to its own port-forward.
    """
    if hasattr(method, "prometheus_port"):
        method.prometheus_port = slot.prometheus_port


def _run_matrix_parallel(
    methods_names: list,
    workloads: list,
    replicates: int,
    duration_s: int,
    complexity: int,
    cooldown_s: int,
    output_dir: Path,
    workers: int,
    cluster_prefix: str,
    base_frontend_port: int,
    base_prometheus_port: int,
    stagger_seconds: int,
    start_replicate: int = 1,
) -> dict[str, Any]:
    """
    Parallel branch — N workers, each owning one kind cluster.

    Per-worker isolation:
      - Own kind context (set via methods.set_thread_kube_context — every
        kubectl/controller-spawn call in this thread targets that cluster).
      - Own frontend host port (workload generator targets only this slot).
      - Own Prometheus port-forward (metrics collection scoped to this slot).

    Required setup BEFORE this runs:
      python setup_parallel_clusters.py <N>
    """
    slots = make_slots(
        workers,
        cluster_prefix=cluster_prefix,
        base_frontend_port=base_frontend_port,
        base_prometheus_port=base_prometheus_port,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not check_disk_space(output_dir):
        return {"status": "low_disk_space", "runs_completed": 0, "runs_failed": 0, "runs": []}

    # Pre-flight every slot — fail fast if a cluster wasn't provisioned
    for slot in slots:
        # Probe the cluster on its own context
        result = subprocess.run(
            ["kubectl", f"--context={slot.kube_context}",
             "get", "ns", NAMESPACE, "-o", "name"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            logger.error(
                "Slot %s pre-flight FAILED: namespace %s not reachable on %s",
                slot.label(), NAMESPACE, slot.kube_context,
            )
            return {
                "status": "cluster_not_ready",
                "missing_slot": slot.label(),
                "runs_completed": 0, "runs_failed": 0, "runs": [],
            }
        if not check_frontend_reachable(slot.frontend_url):
            return {
                "status": "frontend_not_reachable",
                "missing_slot": slot.label(),
                "runs_completed": 0, "runs_failed": 0, "runs": [],
            }
        logger.info("Slot %s pre-flight OK (frontend=%s, prom=%s)",
                    slot.label(), slot.frontend_url, slot.prometheus_url)

    # Build the cell queue
    cells: queue.Queue = queue.Queue()
    total_cells = 0
    for m_spec in methods_names:
        for wl in workloads:
            for rep in range(start_replicate, start_replicate + replicates):
                cells.put((m_spec, wl, rep))
                total_cells += 1

    logger.info("PARALLEL: %d cells across %d workers (~%d cells/worker)",
                 total_cells, workers, (total_cells + workers - 1) // workers)

    run_log = RunLog(output_dir / "run_log.csv")
    run_summaries: list[dict] = []
    summaries_lock = threading.Lock()
    completed = [0]
    failed = [0]
    counters_lock = threading.Lock()
    terminated = threading.Event()

    def handle_sigint(sig, frame):
        logger.warning("\nSIGINT received — letting active workers drain...")
        terminated.set()

    original_handler = signal.signal(signal.SIGINT, handle_sigint)

    def worker(slot: WorkerSlot, stagger_delay: int):
        # Pin this thread to its slot's kube context. Every kubectl()
        # and controller-spawn in this thread now targets slot.kube_context.
        set_thread_kube_context(slot.kube_context)

        if stagger_delay > 0:
            logger.info("[%s] stagger sleep %ds", slot.label(), stagger_delay)
            for _ in range(stagger_delay):
                if terminated.is_set():
                    return
                time.sleep(1)

        # Per-slot Prometheus port-forward
        pf = PortForward(
            namespace=PROMETHEUS_NS,
            service=PROMETHEUS_SVC,
            local_port=slot.prometheus_port,
            remote_port=PROMETHEUS_PORT,
            kube_context=slot.kube_context,
        )
        pf.start()

        try:
            while not terminated.is_set():
                try:
                    m_spec, workload, rep = cells.get_nowait()
                except queue.Empty:
                    return

                # Build a fresh method instance per cell — main's get_method
                # already returns fresh ConfScale/Predictive/etc. instances.
                try:
                    method = _resolve_method_spec(m_spec)
                except (KeyError, TypeError) as e:
                    logger.error("[%s] bad method spec %r: %s — skipping",
                                  slot.label(), m_spec, e)
                    continue

                _apply_slot_to_method(method, slot)

                logger.info(
                    "[%s] starting cell: method=%s workload=%s rep=%d",
                    slot.label(), method.name, workload, rep,
                )
                result = execute_single_run(
                    method=method,
                    workload_pattern=workload,
                    replicate=rep,
                    duration_s=duration_s,
                    complexity=complexity,
                    frontend_url=slot.frontend_url,
                    prometheus_url=slot.prometheus_url,
                    output_dir=output_dir,
                )
                summary = result.get("summary", {})
                run_log.log(
                    run_id=result["run_id"],
                    worker_id=slot.worker_id,
                    method=result["method"],
                    workload=result["workload"],
                    replicate=result["replicate"],
                    status=result["status"],
                    duration_s=result.get("duration_s", duration_s),
                    start_time=result.get("start_time", ""),
                    end_time=result.get("end_time", ""),
                    slo_violation_rate=summary.get("slo_violation_rate", ""),
                    p95_ms=summary.get("p95_ms", ""),
                    mean_replicas=summary.get("mean_replicas", ""),
                    total_requests=summary.get("total_requests", ""),
                    error_rate=summary.get("error_rate", ""),
                    error_message=result.get("error_message", ""),
                )
                with summaries_lock:
                    run_summaries.append(result)
                with counters_lock:
                    if result["status"] == "ok":
                        completed[0] += 1
                    else:
                        failed[0] += 1
                    done = completed[0] + failed[0]
                    logger.info(
                        "[%s] cell done (%d/%d total): %s",
                        slot.label(), done, total_cells, result["status"],
                    )

                if not terminated.is_set():
                    time.sleep(cooldown_s)
        finally:
            pf.stop()
            clear_thread_kube_context()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(worker, slot, i * stagger_seconds)
                for i, slot in enumerate(slots)
            ]
            for f in concurrent.futures.as_completed(futures):
                exc = f.exception()
                if exc:
                    logger.error("Worker crashed: %s", exc)
    finally:
        signal.signal(signal.SIGINT, original_handler)
        run_log.close()

    logger.info("=" * 60)
    logger.info(
        "PARALLEL MATRIX COMPLETE: %d completed, %d failed, %d total cells",
        completed[0], failed[0], total_cells,
    )
    logger.info("Run log: %s", output_dir / "run_log.csv")

    return {
        "status": "complete" if not terminated.is_set() else "terminated",
        "runs_completed": completed[0],
        "runs_failed": failed[0],
        "total_cells": total_cells,
        "workers": workers,
        "runs": run_summaries,
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="P3 Experiment Matrix Orchestrator"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Path to matrix.yaml (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the matrix plan without executing",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=("Parallel workers (default 1 = serial). "
              "When > 1, expects N kind clusters from setup_parallel_clusters.py."),
    )
    parser.add_argument(
        "--cluster-prefix", default=DEFAULT_CLUSTER_PREFIX,
        help=f"Cluster name prefix (default: {DEFAULT_CLUSTER_PREFIX})",
    )
    parser.add_argument(
        "--base-frontend-port", type=int, default=DEFAULT_BASE_FRONTEND_PORT,
        help=f"Host port base for frontend (default: {DEFAULT_BASE_FRONTEND_PORT})",
    )
    parser.add_argument(
        "--base-prometheus-port", type=int, default=DEFAULT_BASE_PROMETHEUS_PORT,
        help=f"Host port base for Prometheus (default: {DEFAULT_BASE_PROMETHEUS_PORT})",
    )
    parser.add_argument(
        "--stagger-seconds", type=int, default=30,
        help="Per-worker startup delay to decorrelate spikes (default: 30)",
    )

    # Single-cell mode
    parser.add_argument("--method", help="Single method name (bypass config)")
    parser.add_argument("--workload", help="Single workload pattern (bypass config)")
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--duration", type=int, default=3600,
                        help="Duration per run in seconds (default: 3600)")
    parser.add_argument("--complexity", type=int, default=50000)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)

    if args.method and args.workload:
        # Single-cell mode
        method = get_method(args.method)
        logger.info("Single-cell mode: %s × %s × rep %d", args.method, args.workload, args.replicate)

        prometheus_pf = PortForward(
            namespace=PROMETHEUS_NS, service=PROMETHEUS_SVC,
            local_port=PROMETHEUS_PORT, remote_port=PROMETHEUS_PORT,
        )
        prometheus_pf.start()

        try:
            result = execute_single_run(
                method=method,
                workload_pattern=args.workload,
                replicate=args.replicate,
                duration_s=args.duration,
                complexity=args.complexity,
                frontend_url=f"http://localhost:{FRONTEND_NODEPORT}",
                prometheus_url=f"http://localhost:{PROMETHEUS_PORT}",
                output_dir=output_dir,
            )
            print(json.dumps(result, indent=2, default=str))
        finally:
            prometheus_pf.stop()
    else:
        # Matrix mode
        result = run_matrix(
            config_path=args.config,
            output_dir=output_dir,
            dry_run=args.dry_run,
            workers=args.workers,
            cluster_prefix=args.cluster_prefix,
            base_frontend_port=args.base_frontend_port,
            base_prometheus_port=args.base_prometheus_port,
            stagger_seconds=args.stagger_seconds,
        )
        if not args.dry_run:
            print(json.dumps({k: v for k, v in result.items() if k != "runs"},
                             indent=2, default=str))


if __name__ == "__main__":
    main()
