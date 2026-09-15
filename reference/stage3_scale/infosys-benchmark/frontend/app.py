"""InfoSys Benchmark — Frontend (API Gateway) Receives HTTP traffic, fans out to downstream services, measures SLO latency."""

if True:
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read README.md#cluster-runs; "
                         "local demo: python -m confscale demo")

from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
import requests
import time
import os

app = Flask(__name__)
metrics = PrometheusMetrics(app)
metrics.info("frontend", "InfoSys Benchmark API Gateway", version="1.0")

latency = metrics.histogram(
    "frontend_latency_seconds", "Request latency (SLO measured here)",
    labels={"path": lambda: request.path}
)

PROCESSOR_URL = os.environ.get("PROCESSOR_URL", "http://processor.infosys-benchmark.svc.cluster.local:8080")
DOWNSTREAM_TIMEOUT = float(os.environ.get("DOWNSTREAM_TIMEOUT", 5.0))


@app.route("/api/process")
@latency
def api_process():
    """Fan-out to processor service."""
    complexity = request.args.get("n", "50000")
    items = request.args.get("items", "3")

    t0 = time.time()
    try:
        resp = requests.get(
            f"{PROCESSOR_URL}/process?n={complexity}&items={items}",
            timeout=DOWNSTREAM_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return jsonify({"error": "downstream_timeout", "elapsed_ms": (time.time() - t0) * 1000}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e), "elapsed_ms": (time.time() - t0) * 1000}), 502

    elapsed_ms = (time.time() - t0) * 1000
    data["frontend_elapsed_ms"] = elapsed_ms
    return jsonify(data)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "frontend"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
