"""
InfoSys Benchmark — Processor
Moderate-CPU service: calls compute-worker (CPU-bound) and cache (I/O-bound),
then aggregates results. Represents a realistic microservice that orchestrates
downstream calls.
"""

# Local artifact reference entrypoint; cluster behavior is unverified.
if True:
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")

from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
import requests
import time
import os

app = Flask(__name__)
metrics = PrometheusMetrics(app)
metrics.info("processor", "InfoSys Benchmark Processor", version="1.0")

latency = metrics.histogram(
    "processor_latency_seconds", "Request latency",
    labels={"path": lambda: request.path}
)

COMPUTE_URL = os.environ.get("COMPUTE_URL", "http://compute-worker.infosys-benchmark.svc.cluster.local:8080")
CACHE_URL = os.environ.get("CACHE_URL", "http://cache.infosys-benchmark.svc.cluster.local:6379")
DOWNSTREAM_TIMEOUT = float(os.environ.get("DOWNSTREAM_TIMEOUT", 5.0))

# Note: cache is Redis, so we use a TCP check pattern (the cache service
# doesn't speak HTTP). We'll access it through a simple sidecar or use
# the Redis protocol directly. For this benchmark, we'll use a lightweight
# Python Redis client if available, or fall back to a simple HTTP proxy.
try:
    import redis
    _redis_client = redis.Redis(host="cache.infosys-benchmark.svc.cluster.local", port=6379,
                                 socket_connect_timeout=2, socket_timeout=2,
                                 decode_responses=True)
except ImportError:
    _redis_client = None


@app.route("/process")
@latency
def process():
    """Orchestrate downstream calls: compute (CPU) + cache (I/O)."""
    n = int(request.args.get("n", 50000))
    items = int(request.args.get("items", 3))

    results = {"compute_results": [], "cache_hits": 0, "cache_misses": 0}

    # Call compute-worker multiple times (fan-out to stress CPU)
    for i in range(items):
        try:
            resp = requests.get(
                f"{COMPUTE_URL}/compute?n={n}",
                timeout=DOWNSTREAM_TIMEOUT
            )
            if resp.status_code == 200:
                data = resp.json()
                results["compute_results"].append({
                    "item": i,
                    "primes": data.get("primes", 0),
                    "pod": data.get("pod", "unknown")
                })
        except requests.exceptions.RequestException as e:
            results["compute_results"].append({"item": i, "error": str(e)})

    # Cache lookups (simulates I/O-bound operation)
    for i in range(items):
        key = f"item:{i}:{n}"
        try:
            if _redis_client:
                cached = _redis_client.get(key)
                if cached:
                    results["cache_hits"] += 1
                else:
                    _redis_client.setex(key, 300, str(results["compute_results"][-1].get("primes", 0)))
                    results["cache_misses"] += 1
        except Exception:
            pass  # Cache is best-effort

    return jsonify(results)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "processor"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
