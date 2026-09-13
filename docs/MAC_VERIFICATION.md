# Mac verification handoff

## Current boundary

This package has local Python verification. It has **no successful standalone Mac, model-training, trace-replay or cluster run**. Reference files are preserved to make that next step concrete. A passing local test suite is not evidence that a Kubernetes deployment launches or reproduces a reported number.

Start by copying this entire directory to the Mac and running:

```console
python3 -m confscale verify
python3 -m confscale reproduce --output generated/mac-paper
python3 -m confscale demo --output generated/mac-demo
python3 -m unittest discover -s tests -v
python3 scripts/mac_preflight.py
```

The preflight only inspects the local platform, executable availability and omitted-input locations. It does not invoke Docker, kind, kubectl, a service endpoint or a download. Retain its output with Python version and test receipts. Compare the regenerated table CSVs and receipt values to the Windows results; do not require identical floating-point demo hashes across platforms without checking the actual difference.

## Before live work

Use a dedicated, explicitly selected test cluster. The historical defaults name `kind-p3-experiments` and `infosys-benchmark`, and several study drivers target three worker clusters. Do not point the original driver at an existing unrelated namespace. Inventory contexts and workloads first, record the initial configuration, and choose new output directories. Do not use the excluded parallel bootstrap or recreate a cluster as a troubleshooting shortcut.

Resolve these known preparation items in the artifact copy:

| Item | Source / required check |
|---|---|
| Dependencies | `reference/stage3_scale/requirements.txt` preserves the historical Python 3.12 pins; installation is unverified. Container Dockerfiles also require `prometheus-flask-exporter`, `gunicorn`, and the processor uses `redis`. Freeze the versions actually installed for the rerun. |
| Testbed images | Build the three service Dockerfiles under `reference/stage3_scale/infosys-benchmark/`; manifests expect `infosys-compute-worker:latest`, `infosys-processor:latest`, `infosys-frontend:latest`, plus Redis. Replace mutable tags with recorded digests for a verification run. No image build or pull has been performed here. |
| Kubernetes and metrics | Inspect the manifests before application. PodMonitor resources need their CRD. The collector historically expects a Prometheus pod from kube-prometheus-stack, while a separate `simple-prometheus.yaml` also exists. Choose and record one actual metrics path, pod, endpoint and scrape configuration. |
| Ports | Source defaults include frontend 30080 and Prometheus 9090. Check collisions, explicitly configure forwarding, and verify served-traffic queries rather than assuming a forwarded port is the right source. |
| Models and training inputs | Models and training CSVs are omitted. Restore them from the source hashes in `provenance/omitted_inputs.json`, or retrain and record this as a new model lineage. The runtime expects `reference/stage3_scale/models/{gru,uq}` and trainers expect `reference/stage3_scale/outputs/training_data`. No symlinks are shipped. |
| Study driver paths | Reference study files retain historical `data/p3_runs/`, `src/stage3_scale`, `.venv` and `/tmp` handshake assumptions. Replace these with explicit artifact-relative input/output parameters on the Mac; check every resolved path before launching. `<SOURCE_WORKSPACE>` marks a scrubbed private path, not a valid default. |
| Trace inputs | Alibaba raw and derived series are omitted. Use locally obtained inputs after checking upstream terms. Hash-match the derived series and preserve the locked selections/gates; changing the selection creates a different analysis. |
| Existing defects/limits | Warm-start must remain available. The original live controller recalibrates h0; the per-horizon extension was evaluated on residual streams separately. Do not represent that offline extension as already deployed live. |

## Smallest useful live check

After the preceding path, dependency and context work, use one method, workload B and a short cell before planning a full matrix. The original runner exposes `--method`, `--workload`, `--replicate`, `--duration`, `--complexity`, `--output-dir` and `--dry-run`. It also exposes `--workers`, `--cluster-prefix` and port bases. Read the copied parser and context routing before selecting them: a single-method shortcut does not itself prove a safe context.

Direct reference runtime scripts are disabled by default. Only after reviewing their resolved commands, set `CONFSCALE_ENABLE_REFERENCE_RUNTIME=1` in that dedicated run environment. The load generator and three service apps also stop on import without this opt-in; this covers their top-level traffic loop and the copied Dockerfile/Gunicorn route. For a reviewed service deployment, explicitly propagate the opt-in into the container environment; a host shell setting is not automatically inherited by a Kubernetes Pod. First inspect the runner's `--dry-run` output; this opt-in also allows workload and cluster mutations in the underlying historical code. The environment variable is an intentional execution stop, not a security boundary or validation certificate.

A short cell should establish that the expected model loads, at least one forecast is issued, an interval is matched to its later observation, scaling targets and actual replicas are separately logged, the intended Prometheus source is used, and the controller and owned forwarders stop cleanly. Capture the run log, controller configuration, scale log, metric summary and actual resources before/after. A short cell cannot confirm paper coverage, cost or p95 results.

Then progress separately to (1) the frozen per-service analysis with supplied trace/model inputs, (2) a baseline A–D cell, (3) F/G/H correction and ladder, and (4) the worker-binding experiment. The binding procedure changes the worker CPU limit and requires explicit restoration evidence; its controller-free static-2/static-20 probe must pass before interpreting a latency contrast. Do not treat a newly generated result as identical to frozen paper evidence without documenting tolerances and causes of differences.

Record each completed stage as a new receipt with command, environment, source/input hashes, outputs, exit state and restoration checks. Leave all original frozen evidence untouched.
