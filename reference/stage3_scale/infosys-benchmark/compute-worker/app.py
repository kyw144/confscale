"""
InfoSys Benchmark — Compute Worker
CPU-bound service (sieve of Eratosthenes). This is the AUTOSCALING TARGET.
The HPA/confidence-aware controller scales this service.
"""

# Local artifact reference entrypoint; cluster behavior is unverified.
if True:
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")

from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
import time
import math
import os

app = Flask(__name__)
metrics = PrometheusMetrics(app)
metrics.info("compute_worker", "CPU-bound compute service for autoscaling tests", version="1.1")

latency = metrics.histogram(
    "compute_latency_seconds", "Request latency",
    labels={"path": lambda: request.path}
)

COMPLEXITY = int(os.environ.get("COMPLEXITY", "50000"))


@app.route("/compute")
@latency
def compute():
    """CPU-bound work — sieve of Eratosthenes."""
    n = int(request.args.get("n", COMPLEXITY))
    t0 = time.time()
    result = sieve(n)
    elapsed = (time.time() - t0) * 1000
    return jsonify({
        "primes": result, "n": n,
        "elapsed_ms": elapsed,
        "pod": os.environ.get("HOSTNAME", "unknown")
    })


@app.route("/compute/heavy")
@latency
def compute_heavy():
    """Heavier CPU work for stress testing (2x default complexity)."""
    n = int(request.args.get("n", COMPLEXITY * 2))
    t0 = time.time()
    result = sieve(n)
    elapsed = (time.time() - t0) * 1000
    return jsonify({
        "primes": result, "n": n,
        "elapsed_ms": elapsed,
        "pod": os.environ.get("HOSTNAME", "unknown")
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "compute-worker", "pod": os.environ.get("HOSTNAME", "unknown")})


def sieve(n: int) -> int:
    """Count primes up to n using sieve of Eratosthenes."""
    if n < 2:
        return 0
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return sum(is_prime)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
