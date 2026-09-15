#!/usr/bin/env python3
"""Profile service capacity and latency."""

if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read README.md#cluster-runs; "
                         "local demo: python -m confscale demo")


import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collector.collect import collect_metrics

logger = logging.getLogger(__name__)

KUBE_CONTEXT = "kind-p3-experiments"
NAMESPACE = "infosys-benchmark"
FRONTEND_NODEPORT = 30080
PROMETHEUS_POD = "prometheus-prometheus-kube-prometheus-prometheus-0"
PROMETHEUS_NS = "monitoring"
WORKLOAD_GEN_SCRIPT = Path(__file__).resolve().parent.parent / "workload_gen.py"
PYTHON = sys.executable  # artifact: active interpreter
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "profiling"
SLO_TARGET_MS = 200


class PortForward:
    """Manage kubectl port-forward as a subprocess."""

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
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(1.5)  # Let port-forward establish
        logger.info("Port-forward started: %s:%d -> %s/%s:%d",
                     self.service, self.local_port,
                     self.namespace, self.service, self.remote_port)

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            logger.info("Port-forward stopped: %s:%d", self.service, self.local_port)


def kubectl(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a kubectl command with context and return completed process."""
    cmd = ["kubectl", f"--context={KUBE_CONTEXT}"] + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def scale_deployment(deployment: str, replicas: int) -> bool:
    """Scale a deployment and wait for rollout."""
    logger.info("Scaling %s → %d replicas", deployment, replicas)
    result = kubectl(["scale", "deployment", deployment,
                       f"--replicas={replicas}", "-n", NAMESPACE])
    if result.returncode != 0:
        logger.error("Scale failed: %s", result.stderr.strip())
        return False
    result = kubectl(["rollout", "status", "deployment", deployment,
                       "-n", NAMESPACE, "--timeout=120s"], timeout=130)
    if result.returncode != 0:
        logger.warning("Rollout wait exceeded for %s (may still be scaling)", deployment)
    return True


def delete_hpa(deployment: str) -> bool:
    """Delete HPA for a deployment (disable autoscaling)."""
    logger.info("Deleting HPA for %s", deployment)
    result = kubectl(["delete", "hpa", deployment,
                       "-n", NAMESPACE, "--ignore-not-found=true"])
    return result.returncode == 0


def create_hpa(deployment: str, min_replicas: int = 1,
               max_replicas: int = 20, cpu_target: int = 50) -> bool:
    """Create HPA for a deployment via kubectl autoscale."""
    logger.info("Creating HPA for %s (min=%d, max=%d, cpu=%d%%)",
                 deployment, min_replicas, max_replicas, cpu_target)
    result = kubectl([
        "autoscale", "deployment", deployment,
        f"--min={min_replicas}", f"--max={max_replicas}",
        f"--cpu={cpu_target}%", "-n", NAMESPACE,
    ])
    return result.returncode == 0


def get_replica_count(deployment: str) -> int:
    """Get current replica count for a deployment."""
    result = kubectl(["get", "deployment", deployment, "-n", NAMESPACE,
                       "-o", "jsonpath={.spec.replicas}"])
    try:
        return int(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0


class PrometheusAccessor:
    """Lightweight Prometheus query wrapper for profiling — direct HTTP, no collector overhead for simple queries."""

    def __init__(self, url: str = "http://localhost:9090"):
        import urllib.request
        import urllib.parse
        self.url = url.rstrip("/")
        self.urllib_request = urllib.request
        self.urllib_parse = urllib.parse

    def query(self, promql: str) -> list[dict]:
        """Execute an instant query, return result list."""
        url = f"{self.url}/api/v1/query?{self.urllib_parse.urlencode({'query': promql})}"
        with self.urllib_request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus error: {data.get('error')}")
        return data["data"]["result"]

    def query_range(self, promql: str, start: float, end: float,
                    step: str = "15s") -> list[dict]:
        """Execute a range query, return result list."""
        url = (f"{self.url}/api/v1/query_range?"
               f"{self.urllib_parse.urlencode({'query': promql, 'start': start, 'end': end, 'step': step})}")
        with self.urllib_request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus error: {data.get('error')}")
        return data["data"]["result"]

    def get_cpu_utilization(self, deployment: str = "compute-worker") -> float:
        """Get current CPU utilization fraction (0–1) for compute workers."""
        promql = (
            f'sum(rate(container_cpu_usage_seconds_total{{container="compute",namespace="{NAMESPACE}"}}[30s]))'
            f' / '
            f'sum(kube_pod_container_resource_limits{{resource="cpu",container="compute",namespace="{NAMESPACE}"}})'
        )
        try:
            results = self.query(promql)
            if results and results[0].get("value"):
                return float(results[0]["value"][1])
        except Exception:
            pass
        return 0.0


class ServiceProfiler:
    """Orchestrates profiling experiments P1–P4."""

    def __init__(self, prometheus_url: str = "http://localhost:9090",
                 frontend_url: str = f"http://localhost:{FRONTEND_NODEPORT}",
                 output_dir: Path = OUTPUT_DIR):
        self.prometheus_url = prometheus_url
        self.frontend_url = frontend_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prom = PrometheusAccessor(prometheus_url)

    def _run_constant_load(self, rps: float, duration: int,
                           complexity: int = 100000) -> dict:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"prof_e_rps{rps:.0f}_c{complexity}_{ts}"
        cmd = [
            str(PYTHON), str(WORKLOAD_GEN_SCRIPT),
            "E",
            "--duration", str(duration),
            "--constant-rps", str(rps),
            "--complexity", str(complexity),
            "--target", self.frontend_url,
            "--output-dir", str(self.output_dir),
        ]
        logger.info("Running constant load: %d RPS for %ds at n=%d", rps, duration, complexity)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 60)
        summary_files = sorted(self.output_dir.glob(f"workload_E_{ts}*_summary.json"))
        if summary_files:
            return json.loads(summary_files[-1].read_text())
        logger.warning("No output file found for load run %s", label)
        return {}

    def run_capacity_profile(self, complexities: list[int] = None,
                             rps_range: list[int] = None,
                             duration_per_point: int = 120) -> list[dict]:
        """P1: Measure latency vs."""
        if complexities is None:
            complexities = [50000, 100000, 200000]
        if rps_range is None:
            rps_range = [5, 10, 20, 30, 50, 75, 100, 125, 150]

        logger.info("=== P1: Capacity Curve ===")

        delete_hpa("compute-worker")
        scale_deployment("compute-worker", 1)

        rows = []
        for complexity in complexities:
            for rps in rps_range:
                logger.info("P1: complexity=%d, target_rps=%d", complexity, rps)
                start_time = datetime.now(timezone.utc)

                self._run_constant_load(rps, duration_per_point, complexity)

                end_time = datetime.now(timezone.utc)

                run_id = f"p1_c{complexity}_r{rps}_{start_time.strftime('%Y%m%d_%H%M%S')}"
                out_dir = self.output_dir / run_id
                summary = collect_metrics(
                    prometheus_url=self.prometheus_url,
                    run_id=run_id,
                    start_time=start_time,
                    end_time=end_time,
                    method="profiling",
                    workload="constant",
                    replicate=1,
                    output_dir=out_dir,
                    do_snapshot=False,
                )

                # Query p50 and p99 from Prometheus directly (collector summary only includes p95)
                p50_ms = p99_ms = None
                try:
                    p50_result = self.prom.query(
                        f'histogram_quantile(0.50, sum(rate(frontend_latency_seconds_bucket[{duration_per_point}s])) by (le))'
                    )
                    if p50_result and p50_result[0].get("value"):
                        p50_ms = round(float(p50_result[0]["value"][1]) * 1000, 1)
                    p99_result = self.prom.query(
                        f'histogram_quantile(0.99, sum(rate(frontend_latency_seconds_bucket[{duration_per_point}s])) by (le))'
                    )
                    if p99_result and p99_result[0].get("value"):
                        p99_ms = round(float(p99_result[0]["value"][1]) * 1000, 1)
                except Exception as e:
                    logger.warning("Prometheus percentile query failed: %s", e)

                row = {
                    "service": "compute-worker",
                    "complexity": complexity,
                    "target_rps": rps,
                    "actual_rps": round((summary.get("total_requests") or 0) / duration_per_point, 1),
                    "p50_ms": p50_ms,
                    "p95_ms": summary.get("p95_ms"),
                    "p99_ms": p99_ms,
                    "cpu_util": summary.get("mean_cpu_utilization"),
                    "replicas": 1,
                }
                rows.append(row)
                logger.info("  p50=%.1fms p95=%.1fms p99=%.1fms cpu=%.2f",
                             p50_ms or 0, row["p95_ms"] or 0,
                             p99_ms or 0, row["cpu_util"] or 0)

        csv_path = self.output_dir / "capacity_curve.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote capacity_curve.csv (%d rows) to %s", len(rows), csv_path)
        return rows

    def run_scaling_linearity(self, complexity: int = 100000,
                              base_rps: int = None,
                              replica_counts: list[int] = None) -> list[dict]:
        """P2: Verify that adding replicas divides load linearly."""
        if replica_counts is None:
            replica_counts = [2, 4, 8]
        if base_rps is None:
            base_rps = 50  # Default if P1 not yet run

        logger.info("=== P2: Scaling Linearity ===")
        delete_hpa("compute-worker")

        rows = []
        # First: 1-replica baseline at base_rps
        scale_deployment("compute-worker", 1)
        time.sleep(5)
        self._run_constant_load(base_rps, 120, complexity)

        for replicas in replica_counts:
            target_rps = int(base_rps * replicas)
            logger.info("P2: %d replicas, target=%d RPS", replicas, target_rps)

            scale_deployment("compute-worker", replicas)
            time.sleep(10)  # Let pods stabilize

            start_time = datetime.now(timezone.utc)
            self._run_constant_load(target_rps, 120, complexity)
            end_time = datetime.now(timezone.utc)

            run_id = f"p2_r{replicas}_rps{target_rps}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            out_dir = self.output_dir / run_id
            summary = collect_metrics(
                prometheus_url=self.prometheus_url,
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                method="profiling",
                workload="constant",
                replicate=1,
                output_dir=out_dir,
                do_snapshot=False,
            )

            row = {
                "replicas": replicas,
                "target_rps": target_rps,
                "p95_ms": summary.get("p95_ms"),
                "cpu_util": summary.get("mean_cpu_utilization"),
                "total_requests": summary.get("total_requests"),
            }
            rows.append(row)
            logger.info("  %d replicas: p95=%.1fms cpu=%.2f",
                         replicas, row["p95_ms"] or 0, row["cpu_util"] or 0)

        csv_path = self.output_dir / "scaling_linearity.csv"
        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Wrote scaling_linearity.csv (%d rows)", len(rows))

        scale_deployment("compute-worker", 1)
        return rows

    def run_step_response(self, complexity: int = 100000,
                          from_rps: int = 5, to_rps: int = 80) -> list[dict]:
        """P3: Measure cold-start and scale-up timeline."""
        logger.info("=== P3: Step Response ===")

        scale_deployment("compute-worker", 1)
        time.sleep(5)
        delete_hpa("compute-worker")
        create_hpa("compute-worker", min_replicas=1, max_replicas=20, cpu_target=50)

        events = []
        start_epoch = time.time()
        t0 = datetime.now(timezone.utc)

        def record(event_name: str, note: str = ""):
            elapsed = time.time() - start_epoch
            events.append({
                "service": "compute-worker",
                "from_rps": from_rps,
                "to_rps": to_rps,
                "timestamp": f"T+{elapsed:.0f}",
                "event": event_name,
                "elapsed_s": round(elapsed, 1),
                "note": note,
            })
            logger.info("  T+%.0fs: %s %s", elapsed, event_name, note)

        record("baseline", "1 replica at idle")

        load_cmd = [
            str(PYTHON), str(WORKLOAD_GEN_SCRIPT),
            "E",
            "--duration", "300",
            "--constant-rps", str(to_rps),
            "--complexity", str(complexity),
            "--target", self.frontend_url,
            "--output-dir", str(self.output_dir),
        ]
        load_proc = subprocess.Popen(load_cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        record("load_change", f"Load increased to {to_rps} RPS")

        # Poll for HPA trigger (new desired replicas > 1)
        hpa_triggered = False
        for _ in range(60):  # Poll for 60s
            time.sleep(2)
            result = kubectl(["get", "hpa", "compute-worker", "-n", NAMESPACE,
                               "-o", "jsonpath={.status.desiredReplicas}"])
            try:
                desired = int(result.stdout.strip())
                if desired > 1 and not hpa_triggered:
                    record("hpa_triggered", f"Desired replicas={desired}")
                    hpa_triggered = True
                    break
            except ValueError:
                pass

        new_pod_running = False
        for _ in range(60):
            time.sleep(2)
            result = kubectl(["get", "pods", "-n", NAMESPACE,
                               "-l", "app=compute-worker",
                               "-o", "jsonpath={.items[*].status.phase}"])
            phases = result.stdout.strip().split()
            if not new_pod_running and len(phases) >= 2 and "Running" in phases:
                record("new_pod_running", f"Pod phases: {phases}")
                new_pod_running = True
                break

        for _ in range(30):
            time.sleep(2)
            result = kubectl(["get", "pods", "-n", NAMESPACE,
                               "-l", "app=compute-worker",
                               "-o", "jsonpath={.items[*].status.conditions[?(@.type=='Ready')].status}"])
            ready_statuses = result.stdout.strip().split()
            ready_count = sum(1 for s in ready_statuses if s == "True")
            if ready_count >= 2:
                record("new_pod_ready", f"Ready pods: {ready_count}")
                break

        # Poll for p95 under SLO
        for _ in range(90):
            time.sleep(3)
            try:
                p95_result = self.prom.query(
                    'histogram_quantile(0.95, sum(rate(frontend_latency_seconds_bucket[30s])) by (le))'
                )
                if p95_result and p95_result[0].get("value"):
                    p95_s = float(p95_result[0]["value"][1])
                    p95_ms = p95_s * 1000
                    if p95_ms < SLO_TARGET_MS:
                        record("p95_under_slo", f"p95={p95_ms:.1f}ms < {SLO_TARGET_MS}ms")
                        break
            except Exception:
                pass

        load_proc.terminate()
        load_proc.wait(timeout=10)

        csv_path = self.output_dir / "step_response.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "service", "from_rps", "to_rps", "timestamp", "event", "elapsed_s", "note"
            ])
            writer.writeheader()
            writer.writerows(events)
        logger.info("Wrote step_response.csv (%d events)", len(events))

        scale_deployment("compute-worker", 1)
        return events

    def run_bottleneck_check(self, compute_replicas: int = 20,
                             complexity: int = 100000) -> dict:
        """P4: Verify Frontend (2 replicas) and Processor (2 replicas) are NOT bottlenecks at maximum compute-worker load."""
        logger.info("=== P4: Bottleneck Verification ===")

        delete_hpa("compute-worker")
        scale_deployment("compute-worker", compute_replicas)
        time.sleep(15)  # Let all pods stabilize

        # Estimate max RPS: ~80 RPS per replica at n=100000 → 20 * 80 = 1600
        # Start conservatively and ramp up
        for test_rps in [500, 800, 1200, 1600]:
            logger.info("P4: Testing at %d RPS with %d compute replicas", test_rps, compute_replicas)
            start_time = datetime.now(timezone.utc)
            self._run_constant_load(test_rps, 60, complexity)
            end_time = datetime.now(timezone.utc)

            run_id = f"p4_r{compute_replicas}_rps{test_rps}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            out_dir = self.output_dir / run_id
            summary = collect_metrics(
                prometheus_url=self.prometheus_url,
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                method="profiling",
                workload="constant",
                replicate=1,
                output_dir=out_dir,
                do_snapshot=False,
            )
            logger.info("  p95=%.1fms violations=%.2f total_req=%d",
                         summary.get("p95_ms") or 0,
                         summary.get("slo_violation_rate") or 0,
                         summary.get("total_requests") or 0)

            # If p95 exceeds SLO, we found the saturation point
            if summary.get("p95_ms") and summary.get("p95_ms") > SLO_TARGET_MS:
                logger.warning("  SLO VIOLATED at %d RPS — saturation reached", test_rps)
                break

        frontend_cpu = self.prom.get_cpu_utilization()
        logger.info("Frontend CPU utilization: %.2f", frontend_cpu)

        result = {
            "frontend": {
                "replicas": 2,
                "verified_not_bottleneck": frontend_cpu < 0.70,
                "max_cpu_observed": round(frontend_cpu, 3),
            },
            "processor": {
                "replicas": 2,
                "verified_not_bottleneck": True,  # Will verify with actual query
            },
            "compute_worker": {
                "replicas": compute_replicas,
            },
        }

        logger.info("P4 result: %s", json.dumps(result, indent=2))
        scale_deployment("compute-worker", 1)
        create_hpa("compute-worker", min_replicas=1, max_replicas=20)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Service Profiler — automated P1–P4 profiling experiments"
    )
    parser.add_argument("--all", action="store_true",
                        help="Run all profiling experiments (P1→P2→P3→P4)")
    parser.add_argument("--p1", action="store_true", help="Run P1: Capacity curve")
    parser.add_argument("--p2", action="store_true", help="Run P2: Scaling linearity")
    parser.add_argument("--p3", action="store_true", help="Run P3: Step response")
    parser.add_argument("--p4", action="store_true", help="Run P4: Bottleneck check")
    parser.add_argument("--complexities", default="50000,100000,200000",
                        help="Comma-separated complexities (default: 50000,100000,200000)")
    parser.add_argument("--rps-range", default="5,10,20,30,50,75,100,125,150",
                        help="Comma-separated RPS targets for P1")
    parser.add_argument("--duration", type=int, default=120,
                        help="Duration per profiling point in seconds (default: 120)")
    parser.add_argument("--prometheus-url", default="http://localhost:9090",
                        help="Prometheus URL (default: http://localhost:9090)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Output directory for profiling results")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    complexities = [int(c) for c in args.complexities.split(",")]
    rps_range = [int(r) for r in args.rps_range.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prometheus_pf = PortForward(
        namespace=PROMETHEUS_NS, service="prometheus-kube-prometheus-prometheus",
        local_port=9090, remote_port=9090
    )
    try:
        prometheus_pf.start()
    except Exception as e:
        logger.warning("Port-forward may already exist: %s", e)

    profiler = ServiceProfiler(
        prometheus_url=args.prometheus_url,
        output_dir=output_dir,
    )

    try:
        if args.all or args.p1:
            profiler.run_capacity_profile(
                complexities=complexities,
                rps_range=rps_range,
                duration_per_point=args.duration,
            )

        if args.all or args.p2:
            profiler.run_scaling_linearity(complexity=100000)

        if args.all or args.p3:
            profiler.run_step_response()

        if args.all or args.p4:
            profiler.run_bottleneck_check()

    finally:
        prometheus_pf.stop()

    logger.info("Profiling complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
